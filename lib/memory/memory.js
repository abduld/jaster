/// <reference path="../ref.ts" />
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
define(["require", "exports"], function (require, exports) {
    var HostMemoryManager = (function (_super) {
        __extends(HostMemoryManager, _super);
        function HostMemoryManager() {
            _super.call(this, AddressSpace.Host);
        }
        return HostMemoryManager;
    })(MemoryManager);
    exports.HostMemoryManager = HostMemoryManager;
    var GlobalMemoryManager = (function (_super) {
        __extends(GlobalMemoryManager, _super);
        function GlobalMemoryManager() {
            _super.call(this, AddressSpace.Global);
        }
        return GlobalMemoryManager;
    })(MemoryManager);
    exports.GlobalMemoryManager = GlobalMemoryManager;
    var SharedMemoryManager = (function (_super) {
        __extends(SharedMemoryManager, _super);
        function SharedMemoryManager() {
            _super.call(this, AddressSpace.Shared);
        }
        return SharedMemoryManager;
    })(MemoryManager);
    exports.SharedMemoryManager = SharedMemoryManager;
});
