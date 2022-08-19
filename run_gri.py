import sys

from stuff.griffiths_solver_vunedited import translator

if __name__ == "__main__":
    print(sys.argv)
    translator.run(sys.argv[1])