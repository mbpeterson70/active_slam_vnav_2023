from active_slam.test_pythonsubpackage1 import bar_fn, baz_fn

def main():
	print(" executing main() in foo.py")
	bar_fn()
	baz_fn()

if __name__ == '__main__':
    main()
