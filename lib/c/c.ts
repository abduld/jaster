/// <reference path="./type/type.ts" />

module lib {
    export module utils { }
    export module c {
        import type = lib.c.type;
    }
}