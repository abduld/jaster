var internal;
(function (internal) {
    var type;
    (function (type) {
        var detail;
        (function (detail) {
            var IntegerTraits = (function () {
                function IntegerTraits() {
                    this.is_integer = function () {
                        return true;
                    };
                    this.is_exact = function () {
                        return true;
                    };
                    this.has_infinity = function () {
                        return false;
                    };
                    this.is_modulo = function () {
                        return true;
                    };
                }

                return IntegerTraits;
            })();
            detail.IntegerTraits = IntegerTraits;
            var SignedIntegerTraits = (function () {
                function SignedIntegerTraits() {
                    this.is_signed = function () {
                        return true;
                    };
                }

                return SignedIntegerTraits;
            })();
            detail.SignedIntegerTraits = SignedIntegerTraits;
            var UnsignedIntegerTraits = (function () {
                function UnsignedIntegerTraits() {
                    this.is_signed = function () {
                        return false;
                    };
                }

                return UnsignedIntegerTraits;
            })();
            detail.UnsignedIntegerTraits = UnsignedIntegerTraits;
        })(detail = type.detail || (type.detail = {}));
    })(type = internal.type || (internal.type = {}));
})(internal || (internal = {}));
