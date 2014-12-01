/// <reference path="./lib/ref.ts" />
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
            b["identifier"]("foo");
        };
        Greeter.prototype.stop = function () {
            lib.utils.assert.ok(1 == 1, "test");
            clearTimeout(this.timerToken);
        };
        return Greeter;
    })();
    app.Greeter = Greeter;
})(app || (app = {}));
window.onload = function () {
    var el = document.getElementById('content');
    var greeter = new app.Greeter(el);
    greeter.start();
};
