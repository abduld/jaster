define(["require", "exports"], function (require, exports) {
    function rand(min, max) {
        return min + Math.random() * (max - min);
    }
    return rand;
});
