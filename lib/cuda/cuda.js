var lib;
(function (lib) {
    var cuda;
    (function (cuda) {
        (function (Status) {
            Status[Status["Running"] = 0] = "Running";
            Status[Status["Idle"] = 1] = "Idle";
            Status[Status["Complete"] = 2] = "Complete";
            Status[Status["Stopped"] = 3] = "Stopped";
            Status[Status["Waiting"] = 4] = "Waiting";
        })(cuda.Status || (cuda.Status = {}));
        var Status = cuda.Status;
        var Dim3 = (function () {
            function Dim3(x, y, z) {
                if (y === void 0) {
                    y = 1;
                }
                if (z === void 0) {
                    z = 1;
                }
                this.x = x;
                this.y = y;
                this.z = z;
            }

            Dim3.prototype.flattenedLength = function () {
                return this.x * this.y * this.z;
            };
            Dim3.prototype.dimension = function () {
                if (this.z == 1) {
                    if (this.y == 1) {
                        return 1;
                    }
                    else {
                        return 2;
                    }
                }
                else {
                    return 3;
                }
            };
            return Dim3;
        })();
        cuda.Dim3 = Dim3;
    })(cuda = lib.cuda || (lib.cuda = {}));
})(lib || (lib = {}));
