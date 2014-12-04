var lib;
(function (lib) {
    var utils;
    (function (utils) {
        function rand(min, max) {
            return min + Math.random() * (max - min);
        }

        utils.rand = rand;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
