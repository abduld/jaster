/// <reference path="../../ref.ts" />
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
define(["require", "exports", "./../../utils/utils"], function(require, exports, utils) {
    (function (AddressSpace) {
        AddressSpace[AddressSpace["Shared"] = 0] = "Shared";
        AddressSpace[AddressSpace["Global"] = 1] = "Global";
        AddressSpace[AddressSpace["Host"] = 2] = "Host";
    })(exports.AddressSpace || (exports.AddressSpace = {}));
    var AddressSpace = exports.AddressSpace;
    ;
    var MemoryObject = (function () {
        function MemoryObject(id, addressSpace, data) {
            this.id = id;
            this.addressSpace = addressSpace;
            this.data = data;
        }
        MemoryObject.prototype.get = function (idx) {
            return this.data[idx];
        };
        MemoryObject.prototype.set = function (idx, val) {
            return this.data[idx] = val;
        };
        return MemoryObject;
    })();
    exports.MemoryObject = MemoryObject;
    ;
    var MemoryManager = (function () {
        function MemoryManager(addressSpace) {
            //private memmap: Map<string, MemoryObject> = new Map<string, MemoryObject>();
            this.MB = 1024;
            this.memoryOffset = 0;
            this.TOTAL_MEMORY = 10 * this.MB;
            this.addressSpace = addressSpace;
            this.memory = new ArrayBuffer(this.TOTAL_MEMORY);
        }
        MemoryManager.prototype.malloc = function (n) {
            var buffer = new MemoryObject(utils.guuid(), this.addressSpace, new DataView(this.memory, this.memoryOffset, this.memoryOffset + n));

            //this.memmap.set(buffer.id, buffer);
            this.memoryOffset += n;
            return buffer;
        };
        MemoryManager.prototype.free = function (mem) {
            mem = undefined;
        };
        MemoryManager.prototype.ref = function (obj) {
            return "todo";
        };
        MemoryManager.prototype.deref = function (mem) {
            return mem[0];
        };
        return MemoryManager;
    })();
    exports.MemoryManager = MemoryManager;
    var HostMemoryManager = (function (_super) {
        __extends(HostMemoryManager, _super);
        function HostMemoryManager() {
            _super.call(this, 2 /* Host */);
        }
        return HostMemoryManager;
    })(MemoryManager);
    exports.HostMemoryManager = HostMemoryManager;
    var GlobalMemoryManager = (function (_super) {
        __extends(GlobalMemoryManager, _super);
        function GlobalMemoryManager() {
            _super.call(this, 1 /* Global */);
        }
        return GlobalMemoryManager;
    })(MemoryManager);
    exports.GlobalMemoryManager = GlobalMemoryManager;
    var SharedMemoryManager = (function (_super) {
        __extends(SharedMemoryManager, _super);
        function SharedMemoryManager() {
            _super.call(this, 0 /* Shared */);
        }
        return SharedMemoryManager;
    })(MemoryManager);
    exports.SharedMemoryManager = SharedMemoryManager;
});
//# sourceMappingURL=memory.js.map
