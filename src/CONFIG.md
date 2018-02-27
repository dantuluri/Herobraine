Install 64 bit python 3.6+
Ensure you have vs2014+ vc++ biniaries
grab MALMO from https://github.com/Microsoft/malmo/releases
Dir Structure:
Root Dir/
| Malmo/
| Herobraine/
| | Src/
| | | minecraft-py/
| | | gym-minecraft/
| | | demo/


Open Power Shell as Administrator:

    pip install keyboard pygame gym
    Set-ExecutionPolicy -Scope CurrentUser Unrestricted
    cd $env:HOMEPATH\Malmo-0.31.0-Windows-64bit_Python3.6\scripts
    .\malmo_install.ps1
    cd ../
    git clone git@github.com:MadcowD/Herobraine.git
    cd ./src/minecraft-py/
    python settup.py install
    cd ../gym-minecraft
    python settup.py install
    


Launch a Malmo Client
Start recording script
