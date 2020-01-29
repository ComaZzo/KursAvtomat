import logging
import re

import sys
from typing import List

from base_classes import LexicalAnalyzer, Lexeme, SyntaxAnalyzer, InvalidSyntax, InvalidLexeme, Interpreter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
logger.addHandler(h)


INFINITY = "Inf"
INFINITY_NUM = 17 ** 6


class MyLexicalAnalyzer(LexicalAnalyzer):
    operation_regex = r'[\*\+()]'
    num_regex = r'[A-G0-9]'

    _logger = logger

    def q_0(self, char: str = None):
        if char is None:
            self.finish()

        elif char.isspace():
            self.loop()

        elif re.fullmatch(self.num_regex, char):
            self.append_lexeme(char)
            self.switch_state(1)

        elif re.fullmatch(self.operation_regex, char):
            self.append_lexeme(char)
            self.finish_lexeme(Lexeme.Type.OPERATION, False)
            self.loop()

        else:
            self.finish_error(f'Lexis error: expected [A-G0-9] or operation char, got "{char}".')

    def q_1(self, char: str):
        if char is None:
            self.finish_lexeme(Lexeme.Type.DATA)
            self.finish()

        elif char.isspace():
            self.finish_lexeme(Lexeme.Type.DATA)
            self.switch_state(0)

        elif re.fullmatch(self.num_regex, char):
            self.append_lexeme(char)
            self.switch_state(2)

        elif re.fullmatch(self.operation_regex, char):
            self.finish_lexeme(Lexeme.Type.DATA)
            self.switch_state(0)

        else:
            self.finish_error(f'Lexis error: expected [A-G0-9] or operation char, got "{char}".')

    def q_2(self, char: str):
        if char is None:
            self.finish_lexeme(Lexeme.Type.DATA)
            self.finish()

        elif char.isspace():
            self.finish_lexeme(Lexeme.Type.DATA)
            self.switch_state(0)

        elif re.fullmatch(self.num_regex, char):
            self.append_lexeme(char)
            self.switch_state(3)

        elif re.fullmatch(self.operation_regex, char):
            self.finish_lexeme(Lexeme.Type.DATA)
            self.switch_state(0)

        else:
            self.finish_error(f'Lexis error: expected [A-G0-9] or operation char, got "{char}".')

    def q_3(self, char: str):
        if char is None:
            self.finish_lexeme(Lexeme.Type.DATA)
            self.finish()

        elif char.isspace():
            self.finish_lexeme(Lexeme.Type.DATA)
            self.switch_state(0)

        elif re.fullmatch(self.num_regex, char):
            self.append_lexeme(char)
            self.switch_state(4)

        elif re.fullmatch(self.operation_regex, char):
            self.finish_lexeme(Lexeme.Type.DATA)
            self.switch_state(0)

        else:
            self.finish_error(f'Lexis error: expected [A-G0-9] or operation char, got "{char}".')

    def q_4(self, char: str):
        if char is None:
            self.finish_lexeme(Lexeme.Type.DATA)
            self.finish()

        elif char.isspace():
            self.finish_lexeme(Lexeme.Type.DATA)
            self.switch_state(0)

        elif re.fullmatch(self.num_regex, char):
            self.append_lexeme(char)
            self.switch_state(5)

        elif re.fullmatch(self.operation_regex, char):
            self.finish_lexeme(Lexeme.Type.DATA)
            self.switch_state(0)

        else:
            self.finish_error(f'Lexis error: expected [A-G0-9] or operation char, got "{char}".')

    def q_5(self, char: str):
        if char is None:
            self.finish_lexeme(Lexeme.Type.DATA)
            self.finish()

        elif char.isspace():
            self.finish_lexeme(Lexeme.Type.DATA)
            self.switch_state(0)

        elif re.fullmatch(self.operation_regex, char):
            self.finish_lexeme(Lexeme.Type.DATA)
            self.switch_state(0)

        else:
            self.finish_error(f'Lexis error: expected operation char, got "{char}".')


def convert_to_17(num: int) -> str:
    if num == 0:
        return "0"

    alphabet = '0123456789ABCDEFG'
    result = []
    while num:
        result.append(alphabet[num % 17])
        num //= 17

    if len(result) > 5:
        return INFINITY

    result.reverse()

    return ''.join(result)


class MySyntaxAnalyzer:

    _logger = logger

    lexeme_list: List[Lexeme] = []
    index: int = 0
    current_lex: Lexeme = None
    state_indent: int = 0

    number_stack: List[str] = []

    PLUS = Lexeme(Lexeme.Type.OPERATION, "+")
    MUL = Lexeme(Lexeme.Type.OPERATION, "*")

    def __init__(self):
        self.index = 0
        self.number_stack = []
        self.state_indent = 0

    def analyze(self, lexeme_list: List[Lexeme]):

        self.__init__()

        self.lexeme_list = lexeme_list

        if len(self.lexeme_list) == 0:
            return "0"

        self.log(f"----------------------------------------\n"
                 f"Syntax analyzer\n"
                 f"----------------------------------------")

        self.next_lexeme()
        self.S()

        self.log(self.number_stack[0])

        return self.number_stack[0]

    def log(self, msg, level=logging.INFO):
        self._logger.log(level, msg)

    def log_state(self, msg, switch: int = 1):
        if switch > 0:
            spaces = ''.join(['| ' for _ in range(self.state_indent)])
            self.log(f'{spaces}{msg}')
            self.state_indent += switch
        else:
            self.state_indent += switch
            spaces = ''.join(['| ' for _ in range(self.state_indent)])
            self.log(f'{spaces}{msg}')

    def next_lexeme(self):
        self.current_lex = self.lexeme_list[self.index]
        self.index += 1

    def error(self, msg):
        self.log(f"Error state: {msg}")
        raise InvalidSyntax(f"Error state: {msg}")

    def S(self):
        self.log_state("State S")
        self.E()
        if self.current_lex.type != Lexeme.Type.END:
            self.error(f"Unknown lexeme: {self.current_lex}")
        self.log_state("End of analyzing", -1)

    def E(self):
        self.log_state("State E")
        self.T()
        self.A()
        self.log_state("End of state E", -1)

    def T(self):
        self.log_state("State T")
        self.F()
        self.B()
        self.log_state("End of state T", -1)

    def A(self):
        self.log_state("State A")

        if self.current_lex.value == "+":
            self.next_lexeme()
            self.E()

            num_2 = self.number_stack.pop()
            num_1 = self.number_stack.pop()

            if num_1 == INFINITY or num_2 == INFINITY:
                self.number_stack.append(INFINITY)
                res = INFINITY
            else:
                res = convert_to_17(int(num_1, 17) + int(num_2, 17))

                if len(res) > 5:
                    res = INFINITY

                self.number_stack.append(res)

            self.log_state(f"Added {num_1} to {num_2} = {res}", 0)

        self.log_state("End of state A", -1)

    def F(self):
        self.log_state("State F")

        if self.current_lex.type == Lexeme.Type.DATA:
            self.number_stack.append(self.current_lex.value)
            self.log_state("Read number", 0)
        else:
            if self.current_lex.value != "(":
                self.error(f'Unexpected lexeme: expected "(" or number, got {self.current_lex}.')

            self.log_state("Read opening brace", 0)
            self.next_lexeme()
            self.E()

            if self.current_lex.value != ")":
                self.error(f'Unexpected lexeme: expected ")", got {self.current_lex}.')

            self.log_state("Read closing brace", 0)

        self.next_lexeme()

        self.log_state("End of state F", -1)

    def B(self):
        self.log_state("State B")

        if self.current_lex.value == "*":
            self.next_lexeme()
            self.T()

            num_2 = self.number_stack.pop()
            num_1 = self.number_stack.pop()

            if num_1 == INFINITY or num_2 == INFINITY:
                self.number_stack.append(INFINITY)
                res = INFINITY
            else:
                res = convert_to_17(int(num_1, 17) * int(num_2, 17))

                if len(res) > 5:
                    res = INFINITY

                self.number_stack.append(res)

            self.log_state(f"Multiplied {num_1} by {num_2} = {res}", 0)

        self.log_state("End of state B", -1)


if __name__ == '__main__':
    lexical_al = MyLexicalAnalyzer()

    # lexemes = lexical_al.analyze("U50L50R50D50l01", {"l01": "[25,101]"})
    syntax_al = MySyntaxAnalyzer()
    # syntax_al.analyze(lexemes)

    interpreter = Interpreter(lexical_al, syntax_al, logger, variable_support=False)
    # interpreter.lexical_analyzer._logger = logging.getLogger("sal")

    # ЭТУ ЧАСТЬ МОЖНО МЕНЯТЬ
    # interpreter.variables = {
    #     "var": "1"
    # }
    # interpreter.interpret("var+10")
    interpreter.shell_interpreter()
