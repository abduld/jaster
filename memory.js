var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var System;
(function (System) {
    var guuid = function () {
        var s4 = function () {
            return Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
        };
        return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() + s4() + s4();
    };
    var MemoryObject = (function () {
        function MemoryObject(id, data) {
            this.id = id;
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
    System.MemoryObject = MemoryObject;
    ;
    var MemoryManager = (function () {
        function MemoryManager() {
            this.memmap = new Map();
            this.MB = 1024;
            this.memory_offset = 0;
            this.TOTAL_MEMORY = 10 * this.MB;
            this.memory = new ArrayBuffer(this.TOTAL_MEMORY);
        }
        MemoryManager.prototype.malloc = function (n) {
            var buffer = new MemoryObject(guuid(), new DataView(this.memory, this.memory_offset, this.memory_offset + n));
            this.memmap.set(buffer.id, buffer);
            this.memory_offset += n;
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
    System.MemoryManager = MemoryManager;

    var HostMemoryManager = (function (_super) {
        __extends(HostMemoryManager, _super);
        function HostMemoryManager() {
            _super.call(this);
        }
        return HostMemoryManager;
    })(MemoryManager);
    System.HostMemoryManager = HostMemoryManager;
    var GlobalMemoryManager = (function (_super) {
        __extends(GlobalMemoryManager, _super);
        function GlobalMemoryManager() {
            _super.call(this);
        }
        return GlobalMemoryManager;
    })(MemoryManager);
    System.GlobalMemoryManager = GlobalMemoryManager;

    var SharedMemoryManager = (function (_super) {
        __extends(SharedMemoryManager, _super);
        function SharedMemoryManager() {
            _super.call(this);
        }
        return SharedMemoryManager;
    })(MemoryManager);
    System.SharedMemoryManager = SharedMemoryManager;
})(System || (System = {}));
//# sourceMappingURL=memory.js.map
