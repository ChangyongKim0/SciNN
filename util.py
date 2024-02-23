def printAfterErase(*args):
    print("\r", end="")
    print(*args, end="")


def fillZero(num, length):
    return str(num).rjust(length, "0")
