// vcam-bridge: feeds video frames into the VirtualCam camera (softcam.dll in the same folder).
//
// Usage:
//   vcam-bridge.exe <width> <height>
//       Reads frames from stdin: width*height*3 bytes each, BGR byte order, top row first.
//   vcam-bridge.exe <width> <height> --test-pattern <r> <g> <b> <seconds>
//       Sends a solid colour for the given time (used by the automated Windows test).
//
// Prints one status line at a time on stdout:
//   ready | connected | disconnected | error <reason>
// Exit codes: 0 normal end, 2 camera already in use, 3 bad arguments, 4 softcam.dll missing.

#include <windows.h>
#include <fcntl.h>
#include <io.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using scCamera = void*;
using CreateFn = scCamera(__cdecl*)(int, int, float);
using DeleteFn = void(__cdecl*)(scCamera);
using SendFn = void(__cdecl*)(scCamera, const void*);
using IsConnectedFn = bool(__cdecl*)(scCamera);

static void Status(const char* text)
{
    std::fprintf(stdout, "%s\n", text);
    std::fflush(stdout);
}

static std::wstring ExeFolder()
{
    wchar_t path[MAX_PATH];
    DWORD len = GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring s(path, len);
    size_t slash = s.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : s.substr(0, slash);
}

// Reads exactly `size` bytes; returns false on end of input.
static bool ReadFull(unsigned char* dst, size_t size)
{
    size_t got = 0;
    while (got < size)
    {
        size_t n = std::fread(dst + got, 1, size - got, stdin);
        if (n == 0) return false;
        got += n;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        Status("error usage: vcam-bridge <width> <height> [--test-pattern r g b seconds]");
        return 3;
    }
    const int width = std::atoi(argv[1]);
    const int height = std::atoi(argv[2]);
    if (width <= 0 || height <= 0 || width % 4 != 0 || height % 4 != 0 || width > 3840 || height > 2160)
    {
        Status("error width and height must be positive multiples of 4");
        return 3;
    }

    bool testPattern = argc >= 8 && std::strcmp(argv[3], "--test-pattern") == 0;

    std::wstring dllPath = ExeFolder() + L"\\softcam.dll";
    HMODULE dll = LoadLibraryW(dllPath.c_str());
    if (!dll)
    {
        Status("error softcam.dll not found next to vcam-bridge.exe");
        return 4;
    }
    auto create = reinterpret_cast<CreateFn>(GetProcAddress(dll, "scCreateCamera"));
    auto destroy = reinterpret_cast<DeleteFn>(GetProcAddress(dll, "scDeleteCamera"));
    auto send = reinterpret_cast<SendFn>(GetProcAddress(dll, "scSendFrame"));
    auto isConnected = reinterpret_cast<IsConnectedFn>(GetProcAddress(dll, "scIsConnected"));
    if (!create || !destroy || !send || !isConnected)
    {
        Status("error softcam.dll is missing functions");
        return 4;
    }

    // Framerate 0: deliver every frame as soon as it arrives (the app sets the pace).
    scCamera cam = create(width, height, 0.0f);
    if (!cam)
    {
        Status("error camera already in use by another program");
        return 2;
    }
    Status("ready");

    std::atomic<bool> running{true};
    std::thread watcher([&] {
        bool last = false;
        while (running)
        {
            bool now = isConnected(cam);
            if (now != last)
            {
                Status(now ? "connected" : "disconnected");
                last = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
    });

    const size_t frameBytes = static_cast<size_t>(width) * height * 3;
    std::vector<unsigned char> frame(frameBytes);

    if (testPattern)
    {
        const unsigned char r = static_cast<unsigned char>(std::atoi(argv[4]));
        const unsigned char g = static_cast<unsigned char>(std::atoi(argv[5]));
        const unsigned char b = static_cast<unsigned char>(std::atoi(argv[6]));
        const double seconds = std::atof(argv[7]);
        for (size_t i = 0; i < frameBytes; i += 3)
        {
            frame[i] = b;
            frame[i + 1] = g;
            frame[i + 2] = r;
        }
        auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<long long>(seconds * 1000));
        while (std::chrono::steady_clock::now() < end)
        {
            send(cam, frame.data());
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }
    else
    {
        _setmode(_fileno(stdin), _O_BINARY);
        while (ReadFull(frame.data(), frameBytes))
        {
            send(cam, frame.data());
        }
    }

    running = false;
    watcher.join();
    destroy(cam);
    FreeLibrary(dll);
    return 0;
}
