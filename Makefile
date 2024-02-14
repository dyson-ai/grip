INCLUDE_DIR = include/
COMPILE_ARGS = -std=c++17 -Wall -Wextra -ffunction-sections -fdata-sections -Og -g -D TRACE_ENABLED 
SRC_FILES = src/*.cpp 
name = dep_injection

ros:
	colcon build --symlink-install
install_debian_deps: 
	sudo apt install python3-bloom python3-rosdep fakeroot debhelper dh-python

debian: clean generate build_debian

generate:
	bloom-generate rosdebian && cp scripts/preinst debian
build_debian: 
	fakeroot debian/rules binary
clean:
	rm -rf ./debian
