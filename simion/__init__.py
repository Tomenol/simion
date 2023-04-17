import pathlib

__lib_path__ = pathlib.Path(__file__).parent.resolve()
__simion_path__ = ""

# Read library config file
conf_file = open(__lib_path__ / pathlib.Path("../config.cfg"))
for line in conf_file.readlines():
	if "simion_path=" in ''.join(line.split()):
		__simion_path__ = line.split('"')[1]