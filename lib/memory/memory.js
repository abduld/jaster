/// <reference path="../ref.ts" />
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var lib;
(function (lib) {
    var memory;
    (function (memory) {
        var detail;
        (function (detail) {
            detail.MemoryManager = lib.c.memory.MemoryManager;
        })(detail = memory.detail || (memory.detail = {}));
        memory.AddressSpace = lib.c.memory.AddressSpace;
        var HostMemoryManager = (function (_super) {
            __extends(HostMemoryManager, _super);
            function HostMemoryManager() {
                _super.call(this, 2 /* Host */);
            }
            return HostMemoryManager;
        })(detail.MemoryManager);
        memory.HostMemoryManager = HostMemoryManager;
        var GlobalMemoryManager = (function (_super) {
            __extends(GlobalMemoryManager, _super);
            function GlobalMemoryManager() {
                _super.call(this, 1 /* Global */);
            }
            return GlobalMemoryManager;
        })(detail.MemoryManager);
        memory.GlobalMemoryManager = GlobalMemoryManager;
        var SharedMemoryManager = (function (_super) {
            __extends(SharedMemoryManager, _super);
            function SharedMemoryManager() {
                _super.call(this, 0 /* Shared */);
            }
            return SharedMemoryManager;
        })(detail.MemoryManager);
        memory.SharedMemoryManager = SharedMemoryManager;
    })(memory = lib.memory || (lib.memory = {}));
})(lib || (lib = {}));
//# sourceMappingURL=memory.js.map