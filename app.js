/// <reference path="./lib/ref.ts" />
define(["require", "exports"], function (require, exports) {
    var app;
    (function (app) {
        var Greeter = (function () {
            function Greeter(element) {
                this.element = element;
                this.element.innerHTML += "The time is: ";
                this.span = document.createElement('span');
                this.element.appendChild(this.span);
                this.span.innerText = new Date().toUTCString();
            }
            Greeter.prototype.start = function () {
                var b = lib.ast.types.builders;
                lib.utils.globals.output = lib.ast.importer.cena.fromCena(lib.example.mp1);
                b["identifier"]("foo");
            };
            Greeter.prototype.stop = function () {
                lib.utils.assert.ok(1 == 1, "test");
                clearTimeout(this.timerToken);
            };
            return Greeter;
        })();
        app.Greeter = Greeter;
    })(app = exports.app || (exports.app = {}));
    window.onload = function () {
        var el = document.getElementById('content');
        var greeter = new app.Greeter(el);
        greeter.start();
    };
});
//# sourceMappingURL=app.js.map