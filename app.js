/// <reference path="./lib/ref.ts" />
define(["require", "exports", "./lib/utils/utils"], function (require, exports, utils) {
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
                var _this = this;
                this.timerToken = setInterval(function () { return _this.span.innerHTML = new Date().toUTCString(); }, 500);
            };
            Greeter.prototype.stop = function () {
                utils.assert(1 == 1, "test");
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
});
