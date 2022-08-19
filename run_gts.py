import sys

from stuff.gts import main

if __name__ == "__main__":
    print(sys.argv)
    main.main(sys.argv[1])