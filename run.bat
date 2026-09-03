
@echo off
rem Remove stale build directory to avoid CMakeCache mismatches
if exist build (
	echo Removing stale build directory...
	rmdir /s /q build
)

echo Configuring with CMake (default generator)...
cmake -S . -B build
if errorlevel 1 (
	echo Default configure failed; attempting MinGW generator...
	cmake -S . -B build -G "MinGW Makefiles"
	if errorlevel 1 (
		echo Configure failed. Please install a build tool - Visual Studio or MinGW, or run cmake manually with an explicit generator.
		pause
		exit /b 1
	)
)

echo Building project...
cmake --build build --config Release
if errorlevel 1 (
	echo Build failed.
	pause
	exit /b 1
)

echo Running demo...
if exist build\nn_demo.exe (
	build\nn_demo.exe
	goto :EOF
)
if exist build\Release\nn_demo.exe (
	build\Release\nn_demo.exe
	goto :EOF
)
if exist build\Debug\nn_demo.exe (
	build\Debug\nn_demo.exe
	goto :EOF
)
echo Could not locate nn_demo.exe — check build output.
dir build /s
pause
exit /b 1