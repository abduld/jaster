define(["require", "exports"], function (require, exports) {
    var NotImplementedError = (function () {
        function NotImplementedError(methodName) {
            this.name = "NotImplementedError";
            this.message = methodName + " should be implemented by React";
        }
        return NotImplementedError;
    })();
    var Component = (function () {
        function Component() {
        }
        Component.prototype.getDOMNode = function () {
            throw new NotImplementedError("getDomNode");
        };
        Component.prototype.setState = function (nextState, callback) {
            throw new NotImplementedError("setState");
        };
        Component.prototype.replaceState = function (nextState, callback) {
            throw new NotImplementedError("replaceState");
        };
        Component.prototype.forceUpdate = function (callback) {
            throw new NotImplementedError("forceUpdate");
        };
        Component.prototype.isMounted = function () {
            throw new NotImplementedError("isMounted");
        };
        Component.prototype.setProps = function (nextProps, callback) {
            throw new NotImplementedError("setProps");
        };
        Component.prototype.replaceProps = function (nextProps, callback) {
            throw new NotImplementedError("replaceProps");
        };
        Component.prototype.render = function () {
            return null;
        };
        return Component;
    })();
    exports.Component = Component;
    var ILLEGAL_KEYS = {
        constructor: true,
        refs: true,
        props: true,
        state: true,
        getDOMNode: true,
        setState: true,
        replaceState: true,
        forceUpdate: true,
        isMounted: true,
        setProps: true,
        replaceProps: true
    };
    function createClass(createClass, clazz) {
        var key;
        var spec = {};
        spec.displayName = clazz.prototype.constructor.name;
        for (key in clazz.prototype) {
            if (!ILLEGAL_KEYS[key]) {
                spec[key] = clazz.prototype[key];
            }
        }
        if (spec.componentWillMount !== undefined) {
            var componentWillMount = spec.componentWillMount;
            spec.componentWillMount = function () {
                clazz.apply(this);
                componentWillMount.apply(this);
            };
        }
        else {
            spec.componentWillMount = function () {
                clazz.apply(this);
            };
        }
        return createClass(spec);
    }
    exports.createClass = createClass;
});
//# sourceMappingURL=reactutils.js.map