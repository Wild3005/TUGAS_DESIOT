konfigurasi pada .vscode

{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/include/opencv4/**",
                "/usr/include/tesseract/**"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "gnu++17",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}


# testing
menggunakan 2 terminal 
terminal 1:
mosquitto_sub -h localhost -t license/plate/response

terminal 2:
mosquitto_pub -h localhost -t license/plate -m "start"
