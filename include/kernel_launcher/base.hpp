#pragma once

#ifndef KERNEL_LAUNCHER_JSON
    #define KERNEL_LAUNCHER_JSON (1)
#endif

#ifndef KERNEL_LAUNCHER_HEADERONLY
    #define KERNEL_LAUNCHER_HEADERONLY (0)
#endif

#if KERNEL_LAUNCHER_HEADERONLY
    #if !KERNEL_LAUNCHER_JSON
        #error \
            "Kernel Launcher must be compiled with support for JSON (KERNEL_LAUNCHER_JSON=1) when used in header-only mode (KERNEL_LAUNCHER_HEADERONLY=1)"
    #endif

    #define KERNEL_LAUNCHER_IDENT(x) x
    #define KERNEL_LAUNCHER_XSTR(x)  #x
    #define KERNEL_LAUNCHER_STR(x)   KERNEL_LAUNCHER_XSTR(x)
    #define KERNEL_LAUNCHER_PATH(x, y) \
        KERNEL_LAUNCHER_STR(KERNEL_LAUNCHER_IDENT(x) KERNEL_LAUNCHER_IDENT(y))
    #define KERNEL_LAUNCHER_IMPL(file) KERNEL_LAUNCHER_PATH("../../src/", file)

    #define KERNEL_LAUNCHER_API inline
#else
    #define KERNEL_LAUNCHER_API
#endif
