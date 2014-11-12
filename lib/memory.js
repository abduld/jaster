var MB = 1024;
var TOTAL_MEMORY = 10 * MB;
var _memory = new ArrayBuffer(TOTAL_MEMORY);
var _memory_offset = 0;
function malloc(n) {
    var buffer = new DataView(_memory, _memory_offset, _memory_offset + n);
    _memory_offset += n;
    return buffer;
}
function free(mem) {
    mem = undefined;
}
function ref(obj) {
    return "todo";
}
function deref(mem) {
    return mem[0];
}
//# sourceMappingURL=memory.js.map