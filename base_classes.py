import datetime
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Callable


@dataclass
class Lexeme:
    class Type(Enum):
        DATA = 0
        OPERATION = 1
        VARIABLE = 2
        END = 3
        NON_TERMINAL = 4

    type: Type
    value: str = ''
    data_type: str = ''

    def __hash__(self):
        return hash(repr(self))

    # region Shortcuts

    @property
    def is_variable(self):
        return self.type == self.Type.VARIABLE

    @property
    def is_data(self):
        return self.type == self.Type.DATA

    @property
    def is_operation(self):
        return self.type == self.Type.OPERATION

    @property
    def is_end(self):
        return self.type == self.Type.END

    @property
    def variable_name(self):
        if not self.is_variable:
            raise ValueError("Lexeme is not a variable name")
        return self.value

    @property
    def operation(self):
        if not self.is_operation:
            raise ValueError("Lexeme is not an operation")
        return self.value

    @property
    def number(self):
        if not self.is_data:
            raise ValueError("Lexeme is not a number")
        return self.value

    def replace(self, lexeme_type: Type, buffer: str):
        """ Replace lexeme without recreating it """

        self.type = lexeme_type
        self.value = buffer

    # endregion


# region Exceptions

class AnalyserError(BaseException):
    pass


class InvalidLexis(AnalyserError):
    pass


class MissingVariable(AnalyserError):
    pass


class InvalidSyntax(AnalyserError):
    pass


class InvalidLexeme(AnalyserError):
    pass


class CalculationError(AnalyserError):
    pass

# endregion


class LexicalAnalyzer:
    """
    Represents lexical analyzer state machine

    Every state function must be named following convention: <state_method_prefix><state_number>
    """

    # region Fields

    state_method_prefix = "q_"

    _logger = logging.getLogger(__name__)

    variable_lexeme_replace_type = Lexeme.Type.DATA

    # endregion

    # region Methods

    def __init__(self):
        self._input_string = ''
        self._index = 0
        self._lexemes: List[Lexeme] = []

        self._current_state: int = 0
        self._next_state: int = -1
        self._dont_read: bool = False
        self._is_error: bool = False
        self._error_msg: str = ''
        self._is_finished: bool = False

        self._is_lexeme_end: bool = False
        self._lexeme_buffer = ''
        self._lexeme_type: Optional[Lexeme.Type] = None

    def analyze(self, input_string: str, variables: Dict[str, str] = None, replace_variables: bool = True) -> List[Lexeme]:
        """ Starting function """

        self.__init__()

        start_timestamp = datetime.datetime.now()
        self.log(f"----------------------------------------\n"
                 f"Lexical analyzer\n"
                 f"----------------------------------------")
        self.log(f"----------------------------------------\n"
                 f"Analysis started at [{start_timestamp}]\n"
                 f"----------------------------------------", logging.DEBUG)
        self._input_string = input_string

        while not self._is_finished:
            state_name = f"{self.state_method_prefix}{self._current_state}"
            if state_name not in dir(self):
                raise RuntimeError(f"STATE FUNCTION {state_name} IS NOT DEFINED.")

            state_func = getattr(self, state_name)
            if not callable(state_func):
                raise RuntimeError(f"INTERNAL FIELD {state_name} MUST BE A STATE FUNCTION")

            if self._dont_read:
                char = self._current_char()
                self._dont_read = False
            else:
                char = self._next_char()

            state_func(char)

            if self._is_error:
                self.log(f'Transition:\tState {self._current_state}\t-- "{char}" -->\tERROR STATE.')
                break

            if not self._is_finished:
                assert self._next_state != -1, f"Next state is not specified in state {state_name}."
                if self._current_state != self._next_state:
                    self.log(f'Transition:\tState {self._current_state}\t-- "{char}" -->\tState {self._next_state}.')
                else:
                    self.log(f'Loop:\t\tState {self._current_state}\t-- "{char}" -->\tState {self._next_state}.')
                self._current_state = self._next_state
                self._next_state = -1

            if self._is_lexeme_end:
                assert self._lexeme_type is not None, f"Lexeme type is not specified in state {state_name}."
                self._lexemes.append(Lexeme(self._lexeme_type, self._lexeme_buffer))
                self._is_lexeme_end = False
                self._lexeme_buffer = ''
                self.log(f'Scanned new lexeme:\t{self._lexemes[-1]}')

        if self._is_error:
            self.log(f"Error: {self._error_msg}")
            raise InvalidLexis(f"Lexical analyzer found error at {self._index} character.")

        stop_timestamp = datetime.datetime.now()
        self.log(f"----------------------------------------\n"
                 f"Analysis ended at [{stop_timestamp}]\n"
                 f"----------------------------------------", logging.DEBUG)

        self.log(f"Got {len(self._lexemes)} lexemes:")
        for lex in self._lexemes:
            self.log(f"\t{lex.type.name}: {lex.value}")
        if replace_variables:
            self._replace_variables(variables)
        self._lexemes.append(Lexeme(Lexeme.Type.END))

        return self._lexemes

    def replace_variable(self, lexeme, variable_value):
        """ Can be overridden to make custom variable replace handling code """

        lexeme.replace(
            self.variable_lexeme_replace_type,
            variable_value
        )

    def _replace_variables(self, variables: Dict[str, str] = None):
        """ Replace variable lexemes with number ones to complete the expression """

        for lexeme in self._lexemes:
            if lexeme.is_variable and variables is not None:
                if lexeme.variable_name in variables.keys():
                    self.replace_variable(lexeme, variables[lexeme.variable_name])
                else:
                    msg = f'Error: Variable "{lexeme.variable_name}" is not defined!'
                    self.log(msg, logging.ERROR)
                    raise MissingVariable(msg)
            elif lexeme.is_variable:
                msg = f'Error: Variable "{lexeme.variable_name}" is not defined!'
                self.log(msg, logging.ERROR)
                raise MissingVariable(msg)

    def _next_char(self):
        """ Get next character of the input string """

        if self._index >= len(self._input_string):
            return None

        ret = self._input_string[self._index]
        self._index += 1
        return ret

    def _current_char(self):
        """ Get last read character """

        return self._input_string[self._index]

    # region In-state data manipulation interface

    def log(self, msg: str, level=logging.INFO):
        self._logger.log(level, msg)

    def loop(self):
        """ Stay in current state """

        self._next_state = self._current_state

    def switch_state(self, num: int):
        """ Declare next state """

        self._next_state = num

    def append_lexeme(self, char: str):
        """ Append char to the lexeme buffer """

        self._lexeme_buffer += char

    def finish_lexeme(self, lexeme_type: Lexeme.Type, return_char: bool = True):
        """ Set lexeme type and lexeme finish flag and return pointer one char back if needed """

        self._lexeme_type = lexeme_type
        self._is_lexeme_end = True

        if return_char:
            self._index -= 1

    def finish(self):
        """ Call it if lexical analysis is finished """

        self._is_finished = True

    def finish_error(self, msg: str = 'Unknown error'):
        """ Exit state machine in the error state """

        self._is_error = True
        self._error_msg = msg

    # endregion

    # endregion


class SyntaxAnalyzer:

    # Both of these can be customised
    state_method_prefix = "q_"
    reduce_rule_method_prefix = 'r_'

    FINAL_STATE = {'__final__': f'{reduce_rule_method_prefix}1'}

    # Defining rules:
    # - outer dict key - transition start (state name)
    # - inner dict key - condition - next lexeme type (as Lexeme.Type) / value (as string)
    # - inner dict value - transition end (state name | reduce rule name)
    # - if state is final (applies reduce rule 1), then mark it as FINAL_STATE (or super().FINAL_STATE)
    #   (replace inner dict)
    rules = {}

    # Optional: can be filled to prettify log
    reduce_rules_text = []

    _logger = logging.getLogger(__name__)

    _current_state: int = 0

    def __init__(self):
        self.index: int = 0
        self.lexeme_stack: List[Lexeme] = []
        self.data_stack: List[int] = []
        self.lexeme_list = None
        self._is_error: bool = False  # Is there are transition to error state
        self._error_msg = ''
        self._is_finished: bool = False  # Is end of analysis reached
        self._lexeme_buffer = None  # Storage for newly-read lexeme

        # Next turn lexeme should not be read and this turn buffered lexeme should not be pushed to stack
        self._no_read = False

        # Non-negative number - if not zero - item from the end of stack that we should read this turn
        self._read_stack_index = 0

    def log(self, msg: str, level=logging.INFO):
        self._logger.log(level, msg)

    def read_lexeme(self) -> Lexeme:
        """ Read and buffer lexeme from stream """

        self._lexeme_buffer = self.lexeme_list[self.index]
        if self.index < len(self.lexeme_list) - 1:
            self.index += 1
        return self._lexeme_buffer

    def push_lexeme(self, lexeme: Lexeme = None):
        """ Push lexeme to the lexeme stack. """

        self.lexeme_stack.append(lexeme)

    def pop_lexeme(self) -> Lexeme:
        return self.lexeme_stack.pop()

    def push_data(self, num):
        self.data_stack.append(num)

    def pop_data(self) -> Any:
        return self.data_stack.pop()

    def _get_next_lexeme(self):
        """ Return next routing lexeme """

        if self._read_stack_index > 0:  # Take lexeme from the stack at certain position
            if self._read_stack_index > len(self.lexeme_stack):
                self._read_stack_index = len(self.lexeme_stack) - 1
            lexeme = self.lexeme_stack[-self._read_stack_index]

            self._no_read = True
            self._read_stack_index -= 1
        elif self._no_read:
            lexeme = self._lexeme_buffer
            self._no_read = False
        else:
            lexeme = self.read_lexeme()

        return lexeme

    def _get_destination(self, lexeme: Lexeme, ruleset: Dict):
        """ Return transition destination string (state name or reduce rule name) """

        # Lexeme explicitly defined
        if lexeme in ruleset.keys():
            return ruleset[lexeme]

        # Lexeme type specified, regardless of value
        elif lexeme.type in ruleset.keys():
            return ruleset[lexeme.type]

        # Lexeme value specified, regardless of type
        elif lexeme.value in ruleset.keys():
            return ruleset[lexeme.value]

        # Special generic cases defined
        elif "__all__" in ruleset.keys():
            return ruleset["__all__"]
        elif "__final__" in ruleset.keys():
            self._is_finished = True
            return ruleset["__final__"]

        # If no transitions defined
        else:
            return None

    def _apply_rule(self, rule_num: int, lexeme: Lexeme):

        rule_str = f"{self.reduce_rule_method_prefix}{rule_num}"
        if rule_str not in dir(self):
            raise RuntimeError(f'ERROR: REDUCE RULE METHOD "{rule_str}" DOES NOT EXIST.')

        reduce_method = getattr(self, rule_str)
        if not callable(reduce_method):
            raise RuntimeError(f'ERROR: INTERNAL FIELD "{rule_str}" MUST BE A REDUCE RULE FUNCTION.')

        reduce_method()

        if len(self.reduce_rules_text) >= rule_num:
            self.log(f'Reduce by rule {rule_num}:  '
                     f'State {self._current_state}'
                     f'  -- {self.reduce_rules_text[rule_num - 1]} -->  '
                     f'State 0.')
        else:
            self.log(f'Reduce by rule {rule_num}:  '
                     f'State {self._current_state}'
                     f'  -- {lexeme.type.name}: {lexeme.value} -->  '
                     f'State 0.')

        self._current_state = 0
        self._read_stack_index = len(self.lexeme_stack)

    def analyze(self, lexeme_list: List[Lexeme]):

        self.__init__()

        def state(num):
            return f"{self.state_method_prefix}{num}"

        start_timestamp = datetime.datetime.now()
        self.log(f"----------------------------------------\n"
                 f"Syntax analyzer\n"
                 f"----------------------------------------")
        self.log(f"----------------------------------------\n"
                 f"Analysis started at [{start_timestamp}]\n"
                 f"----------------------------------------", logging.DEBUG)

        self.lexeme_list = lexeme_list

        while True:
            state_name = state(self._current_state)

            if state_name not in self.rules.keys():
                raise RuntimeError(f'ERROR: STATE "{state_name}" IS NOT DEFINED IN RULES.')

            ruleset = self.rules[state_name]

            lexeme = self._get_next_lexeme()

            destination = self._get_destination(lexeme, ruleset)

            if destination is None:
                msg = (f'Syntax Error: '
                       f'Unexpected lexeme: '
                       f'Transition from "{state_name}" by {repr(lexeme)} is not defined.')
                self.log(msg)
                raise InvalidSyntax(msg)

            # Does destination string refer to the state?
            match = re.fullmatch(f"{self.state_method_prefix}([0-9]+)", destination)
            if match:
                state_num = int(match.group(1))

                if state(state_num) not in self.rules.keys():
                    raise RuntimeError(f'ERROR: STATE "{state(state_num)}" IS NOT DEFINED IN RULES.')

                self.log(f'Transition:  '
                         f'State {self._current_state}'
                         f'  -- {lexeme.type.name}: {lexeme.value} -->  '
                         f'State {state_num}.')
                self._current_state = state_num

                if not self._no_read:
                    self.push_lexeme(self._lexeme_buffer)

                if not self._is_finished:
                    continue
                else:
                    break
            else:
                # If not, does it refer to the reduce rule?
                match = re.fullmatch(f"{self.reduce_rule_method_prefix}([0-9]+)", destination)
                if match:
                    rule_num = int(match.group(1))

                    self._apply_rule(rule_num, lexeme)

                    if not self._is_finished:
                        continue
                    else:
                        break

            # If not either - raise an error
            raise RuntimeError(f'ERROR: RULE DESTINATION "{destination}" UNKNOWN.')

        stop_timestamp = datetime.datetime.now()
        self.log(f"----------------------------------------\n"
                 f"Analysis ended at [{stop_timestamp}]\n"
                 f"----------------------------------------", logging.DEBUG)

        self.log(f'Result = {self.data_stack[0]}')
        return self.data_stack[0]


class Interpreter:

    logger: logging.Logger = None
    lexical_analyzer: LexicalAnalyzer = None
    syntax_analyzer: SyntaxAnalyzer = None
    variables: Dict[str, str] = {}

    help_message = ''

    def __init__(self, lexical_analyzer: LexicalAnalyzer,
                 syntax_analyzer: SyntaxAnalyzer,
                 logger: logging.Logger = None,
                 help_message: str = '',
                 variable_support: bool = True):
        self.lexical_analyzer = lexical_analyzer
        self.syntax_analyzer = syntax_analyzer
        self.help_message = help_message
        self.variable_support = variable_support

        if logger is not None:
            self.logger = logger
            self.lexical_analyzer._logger = logger
            self.syntax_analyzer._logger = logger

    def _interpret(self, line: str) -> int:
        lexeme_stream = self.lexical_analyzer.analyze(line, self.variables)
        result = self.syntax_analyzer.analyze(lexeme_stream)
        return result

    def interpret(self, line: str):
        if not line:
            return

        if '=' in line and self.variable_support:
            var_name, line = line.split('=', 2)
            vn_analyze = self.lexical_analyzer.analyze(var_name.strip(' '), replace_variables=False)
            if len(vn_analyze) != 2 or vn_analyze[0].type != Lexeme.Type.VARIABLE:
                raise AnalyserError('Syntax error: variable name is wrong.')
            result = self._interpret(line)
            self.variables[var_name.strip(' ')] = str(result)
        else:
            result = self._interpret(line)
            print(result)

    def shell_interpreter(self):
        print(self.help_message)
        while True:
            try:
                line = input(">>> ").strip()
                if line == 'set_info':
                    if self.logger is not None:
                        self.logger.setLevel(logging.INFO)
                        print("Set logging level to Info")
                    else:
                        print("Logger is not specified")
                elif line == 'set_warning':
                    if self.logger is not None:
                        self.logger.setLevel(logging.WARNING)
                        print("Set logging level to Warning")
                    else:
                        print("Logger is not specified")
                elif line == 'quit':
                    print("Exitting")
                    break
                else:
                    self.interpret(line)
            except KeyboardInterrupt:
                print("Exitting")
                break
            except AnalyserError as e:
                print(e.args[0])