// Native presenter: a plain Win32 window that blits the latest frame with StretchDIBits.
// It is deliberately simple. Once the renderer runs on the GPU (WebGPU via Dawn natively,
// the browser's WebGPU on the web) presentation moves there; this keeps a window on
// screen while the software path is being brought up.

#include "presenter.h"

#ifdef _WIN32

#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace gpu
{
    static std::mutex g_frameMutex;
    static std::vector<uint32_t> g_frame;
    static uint32_t g_frameWidth = 0, g_frameHeight = 0;
    static std::atomic<uint64_t> g_shown{ 0 };
    static std::atomic<HWND> g_window{ nullptr };
    static std::atomic<bool> g_gpu{ false };

    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        switch (message)
        {
        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC dc = BeginPaint(hwnd, &ps);
            if (g_gpu.load())
            {
                EndPaint(hwnd, &ps);
                return 0;
            }
            RECT client;
            GetClientRect(hwnd, &client);
            {
                std::lock_guard lock(g_frameMutex);
                if (!g_frame.empty())
                {
                    BITMAPINFO info{};
                    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                    info.bmiHeader.biWidth = LONG(g_frameWidth);
                    info.bmiHeader.biHeight = -LONG(g_frameHeight); // top-down
                    info.bmiHeader.biPlanes = 1;
                    info.bmiHeader.biBitCount = 32;
                    info.bmiHeader.biCompression = BI_RGB;
                    SetStretchBltMode(dc, HALFTONE);
                    StretchDIBits(dc, 0, 0, client.right, client.bottom, 0, 0, g_frameWidth, g_frameHeight,
                        g_frame.data(), &info, DIB_RGB_COLORS, SRCCOPY);
                }
                else
                {
                    FillRect(dc, &client, (HBRUSH)GetStockObject(BLACK_BRUSH));
                }
            }
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_ERASEBKGND:
            return 1;
        case WM_CLOSE:
            std::_Exit(0); // the guest has no notion of being closed; just end the process
        default:
            return DefWindowProcW(hwnd, message, wParam, lParam);
        }
    }

    static void WindowThread()
    {
        const HINSTANCE instance = GetModuleHandleW(nullptr);
        WNDCLASSW wc{};
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = instance;
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512)); // IDC_ARROW
        wc.lpszClassName = L"TTT2Window";
        RegisterClassW(&wc);

        RECT rect{ 0, 0, 1280, 720 };
        AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
        HWND hwnd = CreateWindowW(wc.lpszClassName, L"Tekken Tag Tournament 2 (recompiled)", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top, nullptr, nullptr, instance, nullptr);
        g_window = hwnd;

        MSG msg;
        while (GetMessageW(&msg, nullptr, 0, 0) > 0)
        {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    static void EnsureWindow()
    {
        static std::once_flag started;
        std::call_once(started, [] { std::thread(WindowThread).detach(); });
    }

    void* PresenterWindow()
    {
        EnsureWindow();
        while (!g_window.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return g_window.load();
    }

    void PresenterSetGpu(bool gpu) { g_gpu = gpu; }
    void PresenterCountFrame() { ++g_shown; }

    void Present(const uint32_t* bgra, uint32_t width, uint32_t height)
    {
        EnsureWindow();

        {
            std::lock_guard lock(g_frameMutex);
            g_frame.assign(bgra, bgra + size_t(width) * height);
            g_frameWidth = width;
            g_frameHeight = height;
        }
        ++g_shown;
        if (HWND hwnd = g_window.load())
            InvalidateRect(hwnd, nullptr, FALSE);
    }

    uint64_t PresentedToScreen() { return g_shown.load(); }
}

#else

namespace gpu
{
    // Web: presentation goes through the WebGPU canvas (not implemented yet).
    void Present(const uint32_t*, uint32_t, uint32_t) {}
    uint64_t PresentedToScreen() { return 0; }
    void* PresenterWindow() { return nullptr; }
    void PresenterSetGpu(bool) {}
    void PresenterCountFrame() {}
}

#endif
