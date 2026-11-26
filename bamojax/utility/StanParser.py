r""" Translate a Stan program into a bamojax Model.

Typical use:

    model = stan_to_bamojax(stan_code, data=dict(N=..., y=...))

Supported: the data, transformed data, parameters, transformed parameters and model blocks, with
declarations, assignments, `~` statements and for-loops over `~` statements and assignments.
Anything else raises an UnsupportedStanFeature exception.

This functionality was implemented by Claude.AI (Fable 5.2)

"""
import re

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints

from bamojax.base import Model


class UnsupportedStanFeature(Exception):
    pass

#
def multiply(a, b):
    # Stan's * is a matrix product for vectors and matrices, and elementwise if a scalar is involved
    return a @ b if jnp.ndim(a) > 0 and jnp.ndim(b) > 0 else a * b

#
def divide(a, b):
    integers = jnp.issubdtype(jnp.result_type(a), jnp.integer) and jnp.issubdtype(jnp.result_type(b), jnp.integer)
    return a // b if integers else a / b

#
def poisson_log(log_rate):
    return dist.Poisson(rate=jnp.exp(log_rate))

#
def neg_binomial_2_log(log_mean, concentration):
    return dist.NegativeBinomial2(mean=jnp.exp(log_mean), concentration=concentration)

#
def multi_student_t(df, loc, covariance_matrix):
    return dist.MultivariateStudentT(df=df, loc=loc, scale_tril=jnp.linalg.cholesky(covariance_matrix))

#
OPERATORS = {'+': jnp.add, '-': jnp.subtract, '*': multiply, '/': divide, '%': jnp.remainder,
             '.*': jnp.multiply, './': jnp.divide, '^': jnp.power, '.^': jnp.power}

FUNCTIONS = {'exp': jnp.exp, 'log': jnp.log, 'sqrt': jnp.sqrt, 'square': jnp.square, 'abs': jnp.abs, 'fabs': jnp.abs,
             'inv': jnp.reciprocal, 'log1p': jnp.log1p, 'expm1': jnp.expm1, 'pow': jnp.power,
             'sin': jnp.sin, 'cos': jnp.cos, 'tan': jnp.tan, 'tanh': jnp.tanh,
             'log1m': lambda x: jnp.log1p(-x),
             'inv_logit': jax.nn.sigmoid, 'log_inv_logit': jax.nn.log_sigmoid, 'logit': jax.scipy.special.logit,
             'Phi': jax.scipy.stats.norm.cdf, 'softmax': jax.nn.softmax, 'log_softmax': jax.nn.log_softmax,
             'log_sum_exp': jax.nn.logsumexp,
             'sum': jnp.sum, 'mean': jnp.mean, 'cumulative_sum': jnp.cumsum,
             'sd': lambda x: jnp.std(x, ddof=1), 'variance': lambda x: jnp.var(x, ddof=1),
             'dot_product': jnp.dot, 'dot_self': lambda x: jnp.dot(x, x),
             'rows': lambda x: jnp.shape(x)[0], 'cols': lambda x: jnp.shape(x)[1], 'num_elements': jnp.size,
             'rep_vector': lambda x, n: jnp.full(int(n), x), 'rep_row_vector': lambda x, n: jnp.full(int(n), x),
             'rep_array': lambda x, *n: jnp.full([int(i) for i in n], x),
             'rep_matrix': lambda x, n, m: jnp.full((int(n), int(m)), x),
             'to_vector': lambda x: x.T.reshape(-1),
             'diag_matrix': jnp.diag, 'diagonal': jnp.diag, 'inverse': jnp.linalg.inv,
             'cholesky_decompose': jnp.linalg.cholesky,
             'diag_pre_multiply': lambda d, m: d[:, None] * m, 'diag_post_multiply': lambda m, d: m * d,
             'quad_form_diag': lambda m, d: d[:, None] * m * d,
             'multiply_lower_tri_self_transpose': lambda m: m @ m.T}

# Stan distribution -> (numpyro distribution, numpyro names of the Stan arguments)
DISTRIBUTIONS = {'normal': (dist.Normal, ['loc', 'scale']),
                 'std_normal': (dist.Normal, []),
                 'cauchy': (dist.Cauchy, ['loc', 'scale']),
                 'student_t': (dist.StudentT, ['df', 'loc', 'scale']),
                 'double_exponential': (dist.Laplace, ['loc', 'scale']),
                 'logistic': (dist.Logistic, ['loc', 'scale']),
                 'gumbel': (dist.Gumbel, ['loc', 'scale']),
                 'lognormal': (dist.LogNormal, ['loc', 'scale']),
                 'exponential': (dist.Exponential, ['rate']),
                 'gamma': (dist.Gamma, ['concentration', 'rate']),
                 'inv_gamma': (dist.InverseGamma, ['concentration', 'rate']),
                 'chi_square': (dist.Chi2, ['df']),
                 'weibull': (dist.Weibull, ['concentration', 'scale']),
                 'pareto': (dist.Pareto, ['scale', 'alpha']),
                 'beta': (dist.Beta, ['concentration1', 'concentration0']),
                 'beta_proportion': (dist.BetaProportion, ['mean', 'concentration']),
                 'uniform': (dist.Uniform, ['low', 'high']),
                 'von_mises': (dist.VonMises, ['loc', 'concentration']),
                 'bernoulli': (dist.Bernoulli, ['probs']),
                 'bernoulli_logit': (dist.Bernoulli, ['logits']),
                 'binomial': (dist.Binomial, ['total_count', 'probs']),
                 'binomial_logit': (dist.Binomial, ['total_count', 'logits']),
                 'beta_binomial': (dist.BetaBinomial, ['total_count', 'concentration1', 'concentration0']),
                 'poisson': (dist.Poisson, ['rate']),
                 'poisson_log': (poisson_log, ['log_rate']),
                 'neg_binomial': (dist.GammaPoisson, ['concentration', 'rate']),
                 'neg_binomial_2': (dist.NegativeBinomial2, ['mean', 'concentration']),
                 'neg_binomial_2_log': (neg_binomial_2_log, ['log_mean', 'concentration']),
                 'categorical': (dist.Categorical, ['probs']),
                 'categorical_logit': (dist.Categorical, ['logits']),
                 'dirichlet': (dist.Dirichlet, ['concentration']),
                 'multi_normal': (dist.MultivariateNormal, ['loc', 'covariance_matrix']),
                 'multi_normal_cholesky': (dist.MultivariateNormal, ['loc', 'scale_tril']),
                 'multi_normal_prec': (dist.MultivariateNormal, ['loc', 'precision_matrix']),
                 'multi_student_t': (multi_student_t, ['df', 'loc', 'covariance_matrix']),
                 'wishart': (dist.Wishart, ['concentration', 'scale_matrix']),
                 'inv_wishart': (dist.InverseWishart, ['concentration', 'scale_matrix']),
                 'lkj_corr': (dist.LKJ, ['concentration']),
                 'lkj_corr_cholesky': (dist.LKJCholesky, ['concentration'])}

ONE_BASED = ['categorical', 'categorical_logit']  # Stan counts outcomes from 1, numpyro from 0

PLAIN_TYPES = ['int', 'real', 'vector', 'row_vector', 'matrix']
SQUARE_TYPES = ['cov_matrix', 'corr_matrix', 'cholesky_factor_corr', 'cholesky_factor_cov']
OTHER_TYPES = ['simplex', 'ordered', 'positive_ordered', 'unit_vector', 'complex']

# Constrained Stan types are supported when the prior has exactly this support
CONSTRAINTS = {'simplex': constraints.simplex, 'cov_matrix': constraints.positive_definite,
               'corr_matrix': constraints.corr_matrix, 'cholesky_factor_corr': constraints.corr_cholesky}

BLOCKS = ['functions', 'data', 'transformed data', 'parameters', 'transformed parameters', 'model', 'generated quantities']
STATEMENT_KEYWORDS = ['while', 'if', 'target', 'print', 'reject', 'return', 'break', 'continue', 'profile']
EXPRESSION_OPERATORS = ['?', '<', '>', '<=', '>=', '==', '!=', '&&', '||', '!', '\\']
BINARY_LEVELS = [['+', '-'], ['*', '/', '%'], ['.*', './']]  # from loose to tight binding

TOKEN = re.compile(r"""(?P<skip>\s+|//[^\n]*|/\*.*?\*/)
                      |(?P<number>(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)
                      |(?P<name>[A-Za-z_]\w*)
                      |(?P<symbol>\+=|-=|\*=|/=|\.\*|\./|\.\^|<=|>=|==|!=|&&|\|\||[-+*/^%'~=<>()\[\]{},;:?!|\\])""",
                   re.DOTALL | re.VERBOSE)


class StanParser:
    r""" A recursive-descent parser for the supported subset of Stan.

    Expressions and statements are returned as nested tuples, with the kind of the element first, e.g.
    ('op', '+', ('var', 'alpha'), ('num', 1)) or ('sample', lhs, 'normal', [arguments]).

    """

    def __init__(self, code: str):
        self.code = code
        self.tokens = []  # (kind, text, position)
        position = 0
        while position < len(code):
            match = TOKEN.match(code, position)
            if match is None:
                self.unsupported(f"character '{code[position]}'", position)
            if match.lastgroup != 'skip':
                self.tokens.append((match.lastgroup, match.group(), position))
            position = match.end()
        self.tokens.append(('end', 'end of program', len(code)))
        self.current = 0

    #
    def unsupported(self, feature, position=None):
        if position is None:
            position = self.tokens[self.current][2]
        line = self.code.count('\n', 0, position) + 1
        raise UnsupportedStanFeature(f'{feature} (line {line})')

    #
    def peek(self):
        return self.tokens[self.current][1]

    #
    def next(self):
        self.current += 1
        return self.tokens[self.current - 1][1]

    #
    def accept(self, text):
        if self.peek() == text:
            self.current += 1
            return True
        return False

    #
    def expect(self, text):
        if not self.accept(text):
            self.unsupported(f"syntax: expected '{text}' but found '{self.peek()}'")

    #
    def program(self) -> dict:
        r""" Returns a dictionary with the list of statements for each block.

        """
        blocks = {}
        while self.peek() != 'end of program':
            name = self.next()
            if name in ['transformed', 'generated']:
                name = f'{name} {self.next()}'
            if name not in BLOCKS:
                self.unsupported(f"syntax: unknown block '{name}'")
            if name in ['functions', 'generated quantities']:
                self.unsupported(f'{name} block')
            self.expect('{')
            blocks[name] = self.statements()
        return blocks

    #
    def statements(self) -> list:
        statements = []
        while not self.accept('}'):
            statements.append(self.statement())
        return statements

    #
    def statement(self):
        token = self.peek()
        if token in STATEMENT_KEYWORDS:
            self.unsupported(f"'{token}' statement")
        if token in PLAIN_TYPES + SQUARE_TYPES + OTHER_TYPES + ['array']:
            return self.declaration()
        if self.accept('{'):
            return ('block', self.statements())
        if self.accept('for'):
            self.expect('(')
            variable = self.next()
            self.expect('in')
            lower = self.binary()
            if not self.accept(':'):
                self.unsupported('for-loop over the elements of a container')
            upper = self.binary()
            self.expect(')')
            return ('for', variable, lower, upper, self.statement())
        lhs = self.expression()
        if self.accept('~'):
            distribution = self.next()
            self.expect('(')
            arguments = self.expressions(')')
            if self.peek() == 'T':
                self.unsupported('truncation with T[,]')
            self.expect(';')
            return ('sample', lhs, distribution, arguments)
        if self.peek() in ['+=', '-=', '*=', '/=']:
            self.unsupported(f"compound assignment '{self.peek()}'")
        self.expect('=')
        rhs = self.expression()
        self.expect(';')
        return ('assign', lhs, rhs)

    #
    def declaration(self):
        r""" Parses e.g. `vector<lower=0>[K] sigma;` into ('declare', name, type, bounds, dimensions, initial value).

        """
        dimensions = []
        if self.accept('array'):
            self.expect('[')
            dimensions += self.expressions(']')
        type_ = self.next()
        bounds = {}
        if self.accept('<'):
            while True:
                key = self.next()
                self.expect('=')
                bounds[key] = self.binary()
                if not self.accept(','):
                    break
            self.expect('>')
        if self.accept('['):
            dimensions += self.expressions(']')
        if type_ in SQUARE_TYPES and len(dimensions) > 0:
            dimensions.append(dimensions[-1])
        name = self.next()
        if self.accept('['):  # array syntax of older Stan versions: real y[N]
            dimensions = self.expressions(']') + dimensions
        initial = self.expression() if self.accept('=') else None
        self.expect(';')
        return ('declare', name, type_, bounds, dimensions, initial)

    #
    def expressions(self, closing) -> list:
        r""" A comma-separated list of expressions, up to and including the closing bracket.

        """
        items = []
        while not self.accept(closing):
            items.append(self.expression())
            if self.peek() != closing:
                self.expect(',')
        return items

    #
    def expression(self):
        expression = self.binary()
        if self.peek() in EXPRESSION_OPERATORS:
            self.unsupported(f"operator '{self.peek()}'")
        return expression

    #
    def binary(self, level=0):
        if level == len(BINARY_LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek() in BINARY_LEVELS[level]:
            left = ('op', self.next(), left, self.binary(level + 1))
        return left

    #
    def unary(self):
        if self.accept('-'):
            return ('negate', self.unary())
        if self.accept('+'):
            return self.unary()
        if self.peek() == '!':
            self.unsupported("operator '!'")
        base = self.postfix()
        if self.peek() in ['^', '.^']:
            return ('op', self.next(), base, self.unary())
        return base

    #
    def postfix(self):
        expression = self.primary()
        while True:
            if self.accept('['):
                indices = [self.index()]
                while self.accept(','):
                    indices.append(self.index())
                self.expect(']')
                expression = ('index', expression, indices)
            elif self.accept("'"):
                expression = ('transpose', expression)
            else:
                return expression

    #
    def index(self):
        lower = None if self.peek() in [':', ',', ']'] else self.expression()
        if self.accept(':'):
            upper = None if self.peek() in [',', ']'] else self.expression()
            return ('slice', lower, upper)
        if lower is None:
            return ('slice', None, None)
        return lower

    #
    def primary(self):
        kind, text, _ = self.tokens[self.current]
        self.current += 1
        if kind == 'number':
            return ('num', int(text) if text.isdigit() else float(text))
        if kind == 'name':
            if self.accept('('):
                return ('call', text, self.expressions(')'))
            return ('var', text)
        if text == '(':
            expression = self.expression()
            self.expect(')')
            return expression
        if text == '[':
            return ('array', self.expressions(']'))
        if text == '{':
            return ('array', self.expressions('}'))
        self.current -= 1
        self.unsupported(f"syntax: unexpected '{text}'")

    #

#
class StanTranslator:
    r""" Builds a bamojax Model from a parsed Stan program.

    Every `~` statement becomes a stochastic node. Transformed parameters and local variables are
    not nodes; their expressions are evaluated within the link functions of the nodes that use them.

    """

    def __init__(self, blocks: dict, data: dict = None):
        self.data = {}         # values of data and transformed data
        self.parameters = {}   # name -> (type, bounds, shape)
        self.definitions = {}  # name -> (expression, loops), for transformed parameters and local variables
        self.sampling = {}     # name -> (distribution, arguments, loops), for the `~` statements
        self.nodes = {}

        for _, name, type_, _, _, _ in blocks.get('data', []):
            if data is None or name not in data:
                raise ValueError(f"No value provided for data variable '{name}'")
            self.data[name] = jnp.asarray(data[name], dtype=int if type_ == 'int' else float)

        self.collect(blocks.get('transformed data', []), loops=[])
        for name in self.definitions:
            self.data[name] = self.evaluate(('var', name), self.data)
        self.definitions = {}

        for _, name, type_, bounds, dimensions, _ in blocks.get('parameters', []):
            if type_ not in PLAIN_TYPES and type_ not in CONSTRAINTS:
                raise UnsupportedStanFeature(f"parameter '{name}' of type {type_}")
            shape = tuple(int(self.evaluate(d, self.data)) for d in dimensions)
            self.parameters[name] = (type_, bounds, shape)

        self.collect(blocks.get('transformed parameters', []), loops=[])
        self.collect(blocks.get('model', []), loops=[])

    #
    def collect(self, statements: list, loops: list):
        r""" Gathers the assignments and `~` statements, together with the for-loops (variable, size) they are in.

        """
        for statement in statements:
            kind = statement[0]
            if kind == 'block':
                self.collect(statement[1], loops)
            elif kind == 'for':
                _, variable, lower, upper, body = statement
                if int(self.evaluate(lower, self.data)) != 1:
                    raise UnsupportedStanFeature(f"for-loop over '{variable}' that does not start at 1")
                self.collect([body], loops + [(variable, int(self.evaluate(upper, self.data)))])
            elif kind == 'declare':
                if statement[5] is not None:
                    self.definitions[statement[1]] = (statement[5], [])
            elif kind == 'assign':
                name, indexed = self.target(statement[1], loops)
                if name in self.definitions or name in self.data or name in self.parameters:
                    raise UnsupportedStanFeature(f"reassignment of variable '{name}'")
                self.definitions[name] = (statement[2], loops if indexed else [])
            elif kind == 'sample':
                name, indexed = self.target(statement[1], loops)
                if name in self.sampling:
                    raise UnsupportedStanFeature(f"multiple ~ statements for variable '{name}'")
                if name not in self.data and name not in self.parameters:
                    raise UnsupportedStanFeature(f"~ statement for '{name}', which is neither data nor a parameter")
                if len(loops) > 0 and not indexed:
                    raise UnsupportedStanFeature(f"~ statement for '{name}' repeated in a for-loop")
                self.sampling[name] = (statement[2], statement[3], loops)

    #
    def target(self, lhs, loops: list):
        r""" The left-hand side of a statement is a whole variable, or a variable indexed by all enclosing loop variables.

        """
        if lhs[0] == 'var':
            return lhs[1], False
        if lhs[0] == 'index' and lhs[1][0] == 'var' and lhs[2] == [('var', variable) for variable, _ in loops]:
            return lhs[1][1], True
        raise UnsupportedStanFeature('statement for an expression or a partially indexed variable')

    #
    def evaluate(self, expression, env: dict):
        r""" Computes the value of an expression, given the values of variables in env.

        """
        kind = expression[0]
        if kind == 'num':
            return expression[1]
        if kind == 'var':
            name = expression[1]
            if name in env:
                return env[name]
            if name in self.definitions:
                return self.vectorize(*self.definitions[name], env)
            raise UnsupportedStanFeature(f"use of '{name}' where it is undefined or where only data is allowed")
        if kind == 'op':
            return OPERATORS[expression[1]](self.evaluate(expression[2], env), self.evaluate(expression[3], env))
        if kind == 'negate':
            return -self.evaluate(expression[1], env)
        if kind == 'transpose':
            value = self.evaluate(expression[1], env)
            return jnp.swapaxes(value, -1, -2) if jnp.ndim(value) > 1 else value
        if kind == 'array':
            return jnp.array([self.evaluate(item, env) for item in expression[1]])
        if kind == 'call':
            if expression[1] not in FUNCTIONS:
                raise UnsupportedStanFeature(f"function '{expression[1]}'")
            return FUNCTIONS[expression[1]](*[self.evaluate(argument, env) for argument in expression[2]])
        if kind == 'index':
            indices = []
            for index in expression[2]:  # Stan indices start at 1, and slices include the upper bound
                if index[0] == 'slice':
                    lower = None if index[1] is None else int(self.evaluate(index[1], env)) - 1
                    upper = None if index[2] is None else int(self.evaluate(index[2], env))
                    indices.append(slice(lower, upper))
                else:
                    indices.append(self.evaluate(index, env) - 1)
            if sum(not isinstance(i, slice) and jnp.ndim(i) > 0 for i in indices) > 1:
                raise UnsupportedStanFeature('indexing with multiple integer arrays')
            return self.evaluate(expression[1], env)[tuple(indices)]

    #
    def vectorize(self, expression, loops: list, env: dict):
        r""" Evaluates an expression for all values of its for-loop variables at once.

        """
        if len(loops) == 0:
            return self.evaluate(expression, env)
        variable, size = loops[0]

        def body(i):
            return jnp.asarray(self.vectorize(expression, loops[1:], {**env, variable: i}))

        return jax.vmap(body)(jnp.arange(1, size + 1))

    #
    def variables(self, expression):
        r""" Yields the names of all variables in an expression, looking through transformed parameters.

        """
        if isinstance(expression, (tuple, list)):
            if len(expression) > 0 and expression[0] == 'var':
                if expression[1] in self.definitions:
                    yield from self.variables(self.definitions[expression[1]][0])
                else:
                    yield expression[1]
            else:
                for element in expression:
                    yield from self.variables(element)

    #
    def placeholder(self, name: str):
        r""" Returns a valid value for a parameter.

        """
        type_, _, shape = self.parameters[name]
        if type_ in SQUARE_TYPES:
            return jnp.eye(shape[-1])
        if type_ == 'simplex':
            return jnp.ones(shape) / shape[-1]
        return jnp.ones(shape)

    #
    def node(self, name: str):
        r""" Returns the bamojax node of a parameter or data variable; the node and its parents are added to the model if needed.

        """
        if name in self.nodes:
            if self.nodes[name] is None:
                raise UnsupportedStanFeature(f"cyclic dependency between ~ statements involving '{name}'")
            return self.nodes[name]
        if name not in self.sampling:
            self.nodes[name] = self.model.add_node(name, observations=self.data[name])
            return self.nodes[name]

        self.nodes[name] = None  # marks the node as under construction
        stan_distribution, arguments, loops = self.sampling[name]
        if stan_distribution not in DISTRIBUTIONS:
            raise UnsupportedStanFeature(f"distribution '{stan_distribution}'")
        distribution, argument_names = DISTRIBUTIONS[stan_distribution]
        if len(arguments) != len(argument_names):
            raise UnsupportedStanFeature(f"distribution '{stan_distribution}' with {len(arguments)} arguments")

        latent = name in self.parameters
        shape = self.parameters[name][2] if latent else jnp.shape(self.data[name])
        if shape[:len(loops)] != tuple(size for _, size in loops):
            raise UnsupportedStanFeature(f"for-loop over only a part of '{name}'")
        fixed = {}
        if stan_distribution in ['lkj_corr', 'lkj_corr_cholesky']:
            fixed = dict(dimension=shape[-1])

        parents = {}
        for variable in self.variables(arguments):
            if variable in self.parameters or variable in self.data:
                parents[variable] = self.node(variable)

        def link_fn(**parent_values):
            env = {**self.data, **parent_values}
            values = [self.vectorize(argument, loops, env) for argument in arguments]
            return {**fixed, **dict(zip(argument_names, values))}

        # A dry run of the distribution gives its support and shape
        placeholders = {p: self.placeholder(p) if p in self.parameters else self.data[p] for p in parents}
        instance = distribution(**link_fn(**placeholders))

        observations = None
        if latent:
            type_, bounds, _ = self.parameters[name]
            lower, upper = [None if b not in bounds else self.evaluate(bounds[b], self.data) for b in ['lower', 'upper']]
            support = instance.support
            if type_ in CONSTRAINTS:
                if support is not CONSTRAINTS[type_]:
                    raise UnsupportedStanFeature(f"{type_} parameter '{name}' with a {stan_distribution} prior")
            elif lower is not None or upper is not None:
                within_lower = lower is None or jnp.all(getattr(support, 'lower_bound', -jnp.inf) >= lower)
                within_upper = upper is None or jnp.all(getattr(support, 'upper_bound', jnp.inf) <= upper)
                if not (within_lower and within_upper):  # the prior must be truncated to the declared bounds
                    base = distribution

                    def distribution(**kwargs):
                        return dist.TruncatedDistribution(base(**kwargs), low=lower, high=upper)

                    distribution.__name__ = f'Truncated{base.__name__}'

                    try:
                        assert support is constraints.real
                        instance = distribution(**link_fn(**placeholders))
                    except (AssertionError, NotImplementedError):
                        raise UnsupportedStanFeature(f"truncation of the {stan_distribution} prior of '{name}' to its declared bounds")

            # bamojax draws `shape` independent values from the distribution, so only the remaining dimensions are passed
            num_independent = len(shape) - len(instance.shape())
            if num_independent < 0 or shape[num_independent:] != instance.shape():
                raise ValueError(f"Parameter '{name}' has shape {shape}, but its {stan_distribution} prior has shape {instance.shape()}")
            shape = shape[:num_independent]
        else:
            observations = self.data[name]
            if stan_distribution in ONE_BASED:
                observations = observations - 1

        if len(parents) == 0:
            self.nodes[name] = self.model.add_node(name, distribution=instance, observations=observations, shape=shape)
        else:
            self.nodes[name] = self.model.add_node(name, distribution=distribution, observations=observations,
                                                   parents=parents, link_fn=link_fn, shape=shape)
        return self.nodes[name]

    #
    def flat_prior(self, name: str):
        r""" Adds the node of a parameter without a `~` statement: Stan's implicit, improper flat prior on its support.

        """
        type_, bounds, shape = self.parameters[name]
        lower, upper = [None if b not in bounds else self.evaluate(bounds[b], self.data) for b in ['lower', 'upper']]
        if type_ in CONSTRAINTS:
            support = CONSTRAINTS[type_]
        elif lower is None and upper is None:
            support = constraints.real
        elif upper is None:
            support = constraints.greater_than(lower)
        elif lower is None:
            support = constraints.less_than(upper)
        else:
            support = constraints.interval(lower, upper)
        num_independent = len(shape) - support.event_dim
        distribution = dist.ImproperUniform(support, (), shape[num_independent:])
        return self.model.add_node(name, distribution=distribution, shape=shape[:num_independent])

    #
    def build(self, name: str) -> Model:
        self.model = Model(name)
        for parameter in self.parameters:
            if parameter not in self.sampling:
                # raise UnsupportedStanFeature(f"parameter '{parameter}' without a ~ statement (improper flat prior)")
                self.nodes[parameter] = self.flat_prior(parameter)        
        for variable in self.sampling:
            self.node(variable)
        return self.model

    #

#
def stan_to_bamojax(stan_code: str, data: dict = None, name: str = 'Stan model') -> Model:
    r""" Translates a Stan program into a bamojax Model.

    Args:
        stan_code: The Stan program.
        data: A dictionary with a value for each variable in the Stan data block.
        name: The name of the model.
    Returns:
        A bamojax Model with a stochastic node for every `~` statement.

    """
    blocks = StanParser(stan_code).program()
    return StanTranslator(blocks, data).build(name)

#