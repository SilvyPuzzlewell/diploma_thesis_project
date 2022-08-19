from stuff.global_utils.utils import Stack


def eval_prefix(exp):
    if isinstance(exp, str):
        exp_list = exp.split()
    else:
        exp_list = exp
    OPERATORS = {'+', '-', '*', '/', '(', ')', '^'}
    stack = Stack()
    exp_list = exp_list[::-1]
    for symbol in exp_list:
        if symbol not in OPERATORS:
            stack.push(symbol)
        else:
            operand1 = stack.pop()
            operand2 = stack.pop()
            val = eval(f"{operand1} {symbol} {operand2}")
            stack.push(val)
    ret = None
    while not stack.isEmpty():
        ret = stack.pop()

    return ret
#----- https://stackoverflow.com/questions/30067163/evaluating-postfix-in-python
def evalPostfix(exp):
    try:
        if isinstance(exp, str):
            arr = exp.split()
        else:
            arr = exp
        OPERATORS = {'+', '-', '*', '/', '(', ')', '^'}
        s = Stack()
        for symbol in arr:
            if symbol not in OPERATORS:
                result = symbol
            else:
                s2 = s.pop()
                s1 = s.pop()
                result = eval(f"{s1} {symbol} {s2}")
            s.push(result)
        return s.pop()
    except:
        return None







#------ downloaded https://github.com/SAZZAD-AMT/Infix-to-Prefix-Convertion-by-Python
def isOperator(c):
    return (not (c >= 'a' and c <= 'z') and not (c >= '0' and c <= '9') and not (c >= 'A' and c <= 'Z'))


def getPriority(C):
    if (C == '-' or C == '+'):
        return 1
    elif (C == '*' or C == '/'):
        return 2
    elif (C == '^'):
        return 3
    return 0


def infixToPrefix(infix):
    operators = []
    operands = []

    for i in range(len(infix)):

        if (infix[i] == '('):
            operators.append(infix[i])

        elif (infix[i] == ')'):
            while (len(operators) != 0 and (operators[-1] != '(')):
                op1 = operands[-1]
                operands.pop()
                op2 = operands[-1]
                operands.pop()
                op = operators[-1]
                operators.pop()
                tmp = [op, op2, op1]
                operands.append(tmp)
            operators.pop()
        elif (not isOperator(infix[i])):
            operands.append(infix[i] + "")

        else:
            while (len(operators) != 0 and getPriority(infix[i]) <= getPriority(operators[-1])):
                op1 = operands[-1]
                operands.pop()

                op2 = operands[-1]
                operands.pop()

                op = operators[-1]
                operators.pop()

                tmp = [op, op2, op1]
                operands.append(tmp)
            operators.append(infix[i])

    while (len(operators) != 0):
        op1 = operands[-1]
        operands.pop()

        op2 = operands[-1]
        operands.pop()

        op = operators[-1]
        operators.pop()

        tmp = [op, op2, op1]
        operands.append(tmp)
    ret = operands[-1]
    def flatten(test_list):
        if isinstance(test_list, list):
            if len(test_list) == 0:
                return []
            first, rest = test_list[0], test_list[1:]
            return flatten(first) + flatten(rest)
        else:
            return [test_list]
    ret = flatten(ret)
    return ret

#------ downloaded https://www.geeksforgeeks.org/prefix-postfix-conversion/
def prefix2postfix(exp):
    if isinstance(exp, str):
        s = exp.split()
    else:
        s = exp

    stack = []

    operators = {'+', '-', '*', '/', '^'}

    # Reversing the order
    s = s[::-1]

    # iterating through individual tokens
    for i in s:

        # if token is operator
        if i in operators:

            # pop 2 elements from stack
            a = stack.pop()
            b = stack.pop()

            # concatenate them as operand1 +
            # operand2 + operator
            temp = f"{a} {b} {i}"
            stack.append(temp)

        # else if operand
        else:
            stack.append(i)

    # printing final output
    return stack[0]

#x = infixToPrefix(['22', '-', '2', '-', '2'])
#print()
# Convert postfix to Prefix expression

#https://www.geeksforgeeks.org/postfix-prefix-conversion/
def isOperator2(x):
    if x == "+":
        return True

    if x == "-":
        return True

    if x == "/":
        return True

    if x == "*":
        return True

    return False



def postToPre(exp):
    if isinstance(exp, str):
        post_exp= exp.split()
    else:
        post_exp = exp
    s = []

    # length of expression
    length = len(post_exp)

    # reading from right to left
    for i in range(length):

        # check if symbol is operator
        if (isOperator2(post_exp[i])):

            # pop two operands from stack
            op1 = s[-1]
            s.pop()
            op2 = s[-1]
            s.pop()

            # concat the operands and operator
            temp = f"{post_exp[i]} {op2} {op1}"

            # Push string temp back to stack
            s.append(temp)

        # if symbol is an operand
        else:

            # push the operand to the stack
            s.append(post_exp[i])

    ans = ""
    for i in s:
        ans += i
    return ans