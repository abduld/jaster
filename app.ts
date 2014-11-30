/// <reference path="./lib/ref.ts" />

module app {
    import c = lib.c;
    export class Greeter {
        element: HTMLElement;
        span: HTMLElement;
        timerToken: number;

        constructor(element: HTMLElement) {
            this.element = element;
            this.element.innerHTML += "The time is: ";
            this.span = document.createElement('span');
            this.element.appendChild(this.span);
            this.span.innerText = new Date().toUTCString();
        }

        start() {

            var b = lib.ast.types.builders;
            b["identifier"]("foo");
        }

        stop() {
            lib.utils.assert.ok(1 == 1, "test");
            clearTimeout(this.timerToken);
        }

    }
}

window.onload = () => {
    var el = document.getElementById('content');
    var greeter = new app.Greeter(el);
    greeter.start();
};