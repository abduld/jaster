
var MB : number = 1024;
var TOTAL_MEMORY : number = 10 * MB;
var _memory = new ArrayBuffer(TOTAL_MEMORY);
var _memory_offset : number = 0;

function malloc(n : number) {
	var buffer = new DataView(_memory, _memory_offset, _memory_offset + n);
	_memory_offset += n;
	return buffer;
}

function free(mem) {
	mem = undefined;
}

function ref(obj) {
	return "todo"
}

function deref(mem) {
	return mem[0];
}



