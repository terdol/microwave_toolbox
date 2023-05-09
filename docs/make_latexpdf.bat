set PATH=C:\Users\Erdoel\Programlar\MiKTeX\miktex\bin\x64;C:\Users\Erdoel\Programlar\msys64\usr\bin;%PATH%
call make.bat latexpdf
move /Y .\build\latex\spfile.pdf .\build\latex\Programming_Reference.pdf