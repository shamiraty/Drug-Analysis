/*!
 Buttons for DataTables 1.7.1
 ©2016-2021 SpryMedia Ltd - datatables.net/license
*/
(function(e) {
    "function" === typeof define && define.amd ? define(["jquery", "datatables.net"], function(y) { return e(y, window, document) }) : "object" === typeof exports ? module.exports = function(y, w) {
        y || (y = window);
        w && w.fn.dataTable || (w = require("datatables.net")(y, w).$);
        return e(w, y, y.document)
    } : e(jQuery, window, document)
})(function(e, y, w, r) {
    function B(a, b, c) { e.fn.animate ? a.stop().fadeIn(b, c) : (a.css("display", "block"), c && c.call(a)) }

    function C(a, b, c) { e.fn.animate ? a.stop().fadeOut(b, c) : (a.css("display", "none"), c && c.call(a)) }

    function E(a, b) {
        a = new q.Api(a);
        b = b ? b : a.init().buttons || q.defaults.buttons;
        return (new t(a, b)).container()
    }
    var q = e.fn.dataTable,
        I = 0,
        J = 0,
        x = q.ext.buttons,
        t = function(a, b) {
            if (!(this instanceof t)) return function(c) { return (new t(c, a)).container() };
            "undefined" === typeof b && (b = {});
            !0 === b && (b = {});
            Array.isArray(b) && (b = { buttons: b });
            this.c = e.extend(!0, {}, t.defaults, b);
            b.buttons && (this.c.buttons = b.buttons);
            this.s = { dt: new q.Api(a), buttons: [], listenKeys: "", namespace: "dtb" + I++ };
            this.dom = {
                container: e("<" + this.c.dom.container.tag +
                    "/>").addClass(this.c.dom.container.className)
            };
            this._constructor()
        };
    e.extend(t.prototype, {
        action: function(a, b) {
            a = this._nodeToButton(a);
            if (b === r) return a.conf.action;
            a.conf.action = b;
            return this
        },
        active: function(a, b) {
            var c = this._nodeToButton(a);
            a = this.c.dom.button.active;
            c = e(c.node);
            if (b === r) return c.hasClass(a);
            c.toggleClass(a, b === r ? !0 : b);
            return this
        },
        add: function(a, b) {
            var c = this.s.buttons;
            if ("string" === typeof b) {
                b = b.split("-");
                var d = this.s;
                c = 0;
                for (var f = b.length - 1; c < f; c++) d = d.buttons[1 * b[c]];
                c = d.buttons;
                b = 1 * b[b.length - 1]
            }
            this._expandButton(c, a, d !== r, b);
            this._draw();
            return this
        },
        container: function() { return this.dom.container },
        disable: function(a) {
            a = this._nodeToButton(a);
            e(a.node).addClass(this.c.dom.button.disabled).attr("disabled", !0);
            return this
        },
        destroy: function() {
            e("body").off("keyup." + this.s.namespace);
            var a = this.s.buttons.slice(),
                b;
            var c = 0;
            for (b = a.length; c < b; c++) this.remove(a[c].node);
            this.dom.container.remove();
            a = this.s.dt.settings()[0];
            c = 0;
            for (b = a.length; c < b; c++)
                if (a.inst === this) {
                    a.splice(c,
                        1);
                    break
                }
            return this
        },
        enable: function(a, b) {
            if (!1 === b) return this.disable(a);
            a = this._nodeToButton(a);
            e(a.node).removeClass(this.c.dom.button.disabled).removeAttr("disabled");
            return this
        },
        name: function() { return this.c.name },
        node: function(a) {
            if (!a) return this.dom.container;
            a = this._nodeToButton(a);
            return e(a.node)
        },
        processing: function(a, b) {
            var c = this.s.dt,
                d = this._nodeToButton(a);
            if (b === r) return e(d.node).hasClass("processing");
            e(d.node).toggleClass("processing", b);
            e(c.table().node()).triggerHandler("buttons-processing.dt", [b, c.button(a), c, e(a), d.conf]);
            return this
        },
        remove: function(a) {
            var b = this._nodeToButton(a),
                c = this._nodeToHost(a),
                d = this.s.dt;
            if (b.buttons.length)
                for (var f = b.buttons.length - 1; 0 <= f; f--) this.remove(b.buttons[f].node);
            b.conf.destroy && b.conf.destroy.call(d.button(a), d, e(a), b.conf);
            this._removeKey(b.conf);
            e(b.node).remove();
            a = e.inArray(b, c);
            c.splice(a, 1);
            return this
        },
        text: function(a, b) {
            var c = this._nodeToButton(a);
            a = this.c.dom.collection.buttonLiner;
            a = c.inCollection && a && a.tag ? a.tag : this.c.dom.buttonLiner.tag;
            var d = this.s.dt,
                f = e(c.node),
                h = function(m) { return "function" === typeof m ? m(d, f, c.conf) : m };
            if (b === r) return h(c.conf.text);
            c.conf.text = b;
            a ? f.children(a).html(h(b)) : f.html(h(b));
            return this
        },
        _constructor: function() {
            var a = this,
                b = this.s.dt,
                c = b.settings()[0],
                d = this.c.buttons;
            c._buttons || (c._buttons = []);
            c._buttons.push({ inst: this, name: this.c.name });
            for (var f = 0, h = d.length; f < h; f++) this.add(d[f]);
            b.on("destroy", function(m, g) { g === c && a.destroy() });
            e("body").on("keyup." + this.s.namespace, function(m) {
                if (!w.activeElement ||
                    w.activeElement === w.body) { var g = String.fromCharCode(m.keyCode).toLowerCase(); - 1 !== a.s.listenKeys.toLowerCase().indexOf(g) && a._keypress(g, m) }
            })
        },
        _addKey: function(a) { a.key && (this.s.listenKeys += e.isPlainObject(a.key) ? a.key.key : a.key) },
        _draw: function(a, b) {
            a || (a = this.dom.container, b = this.s.buttons);
            a.children().detach();
            for (var c = 0, d = b.length; c < d; c++) a.append(b[c].inserter), a.append(" "), b[c].buttons && b[c].buttons.length && this._draw(b[c].collection, b[c].buttons)
        },
        _expandButton: function(a, b, c, d) {
            var f =
                this.s.dt,
                h = 0;
            b = Array.isArray(b) ? b : [b];
            for (var m = 0, g = b.length; m < g; m++) {
                var n = this._resolveExtends(b[m]);
                if (n)
                    if (Array.isArray(n)) this._expandButton(a, n, c, d);
                    else {
                        var k = this._buildButton(n, c);
                        k && (d !== r && null !== d ? (a.splice(d, 0, k), d++) : a.push(k), k.conf.buttons && (k.collection = e("<" + this.c.dom.collection.tag + "/>"), k.conf._collection = k.collection, this._expandButton(k.buttons, k.conf.buttons, !0, d)), n.init && n.init.call(f.button(k.node), f, e(k.node), n), h++)
                    }
            }
        },
        _buildButton: function(a, b) {
            var c = this.c.dom.button,
                d = this.c.dom.buttonLiner,
                f = this.c.dom.collection,
                h = this.s.dt,
                m = function(p) { return "function" === typeof p ? p(h, k, a) : p };
            b && f.button && (c = f.button);
            b && f.buttonLiner && (d = f.buttonLiner);
            if (a.available && !a.available(h, a)) return !1;
            var g = function(p, l, v, u) {
                u.action.call(l.button(v), p, l, v, u);
                e(l.table().node()).triggerHandler("buttons-action.dt", [l.button(v), l, v, u])
            };
            f = a.tag || c.tag;
            var n = a.clickBlurs === r ? !0 : a.clickBlurs,
                k = e("<" + f + "/>").addClass(c.className).attr("tabindex", this.s.dt.settings()[0].iTabIndex).attr("aria-controls",
                    this.s.dt.table().node().id).on("click.dtb", function(p) {
                    p.preventDefault();
                    !k.hasClass(c.disabled) && a.action && g(p, h, k, a);
                    n && k.trigger("blur")
                }).on("keyup.dtb", function(p) { 13 === p.keyCode && !k.hasClass(c.disabled) && a.action && g(p, h, k, a) });
            "a" === f.toLowerCase() && k.attr("href", "#");
            "button" === f.toLowerCase() && k.attr("type", "button");
            d.tag ? (f = e("<" + d.tag + "/>").html(m(a.text)).addClass(d.className), "a" === d.tag.toLowerCase() && f.attr("href", "#"), k.append(f)) : k.html(m(a.text));
            !1 === a.enabled && k.addClass(c.disabled);
            a.className && k.addClass(a.className);
            a.titleAttr && k.attr("title", m(a.titleAttr));
            a.attr && k.attr(a.attr);
            a.namespace || (a.namespace = ".dt-button-" + J++);
            d = (d = this.c.dom.buttonContainer) && d.tag ? e("<" + d.tag + "/>").addClass(d.className).append(k) : k;
            this._addKey(a);
            this.c.buttonCreated && (d = this.c.buttonCreated(a, d));
            return { conf: a, node: k.get(0), inserter: d, buttons: [], inCollection: b, collection: null }
        },
        _nodeToButton: function(a, b) {
            b || (b = this.s.buttons);
            for (var c = 0, d = b.length; c < d; c++) {
                if (b[c].node === a) return b[c];
                if (b[c].buttons.length) { var f = this._nodeToButton(a, b[c].buttons); if (f) return f }
            }
        },
        _nodeToHost: function(a, b) { b || (b = this.s.buttons); for (var c = 0, d = b.length; c < d; c++) { if (b[c].node === a) return b; if (b[c].buttons.length) { var f = this._nodeToHost(a, b[c].buttons); if (f) return f } } },
        _keypress: function(a, b) {
            if (!b._buttonsHandled) {
                var c = function(d) {
                    for (var f = 0, h = d.length; f < h; f++) {
                        var m = d[f].conf,
                            g = d[f].node;
                        m.key && (m.key === a ? (b._buttonsHandled = !0, e(g).click()) : !e.isPlainObject(m.key) || m.key.key !== a || m.key.shiftKey &&
                            !b.shiftKey || m.key.altKey && !b.altKey || m.key.ctrlKey && !b.ctrlKey || m.key.metaKey && !b.metaKey || (b._buttonsHandled = !0, e(g).click()));
                        d[f].buttons.length && c(d[f].buttons)
                    }
                };
                c(this.s.buttons)
            }
        },
        _removeKey: function(a) {
            if (a.key) {
                var b = e.isPlainObject(a.key) ? a.key.key : a.key;
                a = this.s.listenKeys.split("");
                b = e.inArray(b, a);
                a.splice(b, 1);
                this.s.listenKeys = a.join("")
            }
        },
        _resolveExtends: function(a) {
            var b = this.s.dt,
                c, d = function(g) {
                    for (var n = 0; !e.isPlainObject(g) && !Array.isArray(g);) {
                        if (g === r) return;
                        if ("function" ===
                            typeof g) { if (g = g(b, a), !g) return !1 } else if ("string" === typeof g) {
                            if (!x[g]) throw "Unknown button type: " + g;
                            g = x[g]
                        }
                        n++;
                        if (30 < n) throw "Buttons: Too many iterations";
                    }
                    return Array.isArray(g) ? g : e.extend({}, g)
                };
            for (a = d(a); a && a.extend;) {
                if (!x[a.extend]) throw "Cannot extend unknown button type: " + a.extend;
                var f = d(x[a.extend]);
                if (Array.isArray(f)) return f;
                if (!f) return !1;
                var h = f.className;
                a = e.extend({}, f, a);
                h && a.className !== h && (a.className = h + " " + a.className);
                var m = a.postfixButtons;
                if (m) {
                    a.buttons || (a.buttons = []);
                    h = 0;
                    for (c = m.length; h < c; h++) a.buttons.push(m[h]);
                    a.postfixButtons = null
                }
                if (m = a.prefixButtons) {
                    a.buttons || (a.buttons = []);
                    h = 0;
                    for (c = m.length; h < c; h++) a.buttons.splice(h, 0, m[h]);
                    a.prefixButtons = null
                }
                a.extend = f.extend
            }
            return a
        },
        _popover: function(a, b, c) {
            var d = this.c,
                f = e.extend({
                    align: "button-left",
                    autoClose: !1,
                    background: !0,
                    backgroundClassName: "dt-button-background",
                    contentClassName: d.dom.collection.className,
                    collectionLayout: "",
                    collectionTitle: "",
                    dropup: !1,
                    fade: 400,
                    rightAlignClassName: "dt-button-right",
                    tag: d.dom.collection.tag
                }, c),
                h = b.node(),
                m = function() {
                    C(e(".dt-button-collection"), f.fade, function() { e(this).detach() });
                    e(b.buttons('[aria-haspopup="true"][aria-expanded="true"]').nodes()).attr("aria-expanded", "false");
                    e("div.dt-button-background").off("click.dtb-collection");
                    t.background(!1, f.backgroundClassName, f.fade, h);
                    e("body").off(".dtb-collection");
                    b.off("buttons-action.b-internal")
                };
            !1 === a && m();
            c = e(b.buttons('[aria-haspopup="true"][aria-expanded="true"]').nodes());
            c.length && (h = c.eq(0), m());
            c = e("<div/>").addClass("dt-button-collection").addClass(f.collectionLayout).css("display", "none");
            a = e(a).addClass(f.contentClassName).attr("role", "menu").appendTo(c);
            h.attr("aria-expanded", "true");
            h.parents("body")[0] !== w.body && (h = w.body.lastChild);
            f.collectionTitle && c.prepend('<div class="dt-button-collection-title">' + f.collectionTitle + "</div>");
            B(c.insertAfter(h), f.fade);
            d = e(b.table().container());
            var g = c.css("position");
            "dt-container" === f.align && (h = h.parent(), c.css("width", d.width()));
            if ("absolute" ===
                g) {
                var n = h.position();
                g = e(b.node()).position();
                c.css({ top: g.top + h.outerHeight(), left: n.left });
                n = c.outerHeight();
                var k = d.offset().top + d.height();
                k = g.top + h.outerHeight() + n - k;
                var p = g.top - n,
                    l = d.offset().top;
                g = g.top - n - 5;
                (k > l - p || f.dropup) && -g < l && c.css("top", g);
                g = d.offset().left;
                d = d.width();
                d = g + d;
                n = c.offset().left;
                k = c.width();
                k = n + k;
                p = h.offset().left;
                l = h.outerWidth();
                var v = p + l;
                c.hasClass(f.rightAlignClassName) || c.hasClass(f.leftAlignClassName) || "dt-container" === f.align ? (l = 0, c.hasClass(f.rightAlignClassName) ?
                    (l = v - k, g > n + l && (g -= n + l, d -= k + l, l = g > d ? l + d : l + g)) : (l = g - n, d < k + l && (g -= n + l, d -= k + l, l = g > d ? l + d : l + g))) : (d = h.offset().top, l = 0, l = "button-right" === f.align ? v - k : p - n);
                c.css("left", c.position().left + l)
            } else d = c.height() / 2, d > e(y).height() / 2 && (d = e(y).height() / 2), c.css("marginTop", -1 * d);
            f.background && t.background(!0, f.backgroundClassName, f.fade, h);
            e("div.dt-button-background").on("click.dtb-collection", function() {});
            e("body").on("click.dtb-collection", function(u) {
                var z = e.fn.addBack ? "addBack" : "andSelf",
                    F = e(u.target).parent()[0];
                (!e(u.target).parents()[z]().filter(a).length && !e(F).hasClass("dt-buttons") || e(u.target).hasClass("dt-button-background")) && m()
            }).on("keyup.dtb-collection", function(u) { 27 === u.keyCode && m() });
            f.autoClose && setTimeout(function() { b.on("buttons-action.b-internal", function(u, z, F, K) { K[0] !== h[0] && m() }) }, 0);
            e(c).trigger("buttons-popover.dt")
        }
    });
    t.background = function(a, b, c, d) {
        c === r && (c = 400);
        d || (d = w.body);
        a ? B(e("<div/>").addClass(b).css("display", "none").insertAfter(d), c) : C(e("div." + b), c, function() { e(this).removeClass(b).remove() })
    };
    t.instanceSelector = function(a, b) {
        if (a === r || null === a) return e.map(b, function(h) { return h.inst });
        var c = [],
            d = e.map(b, function(h) { return h.name }),
            f = function(h) {
                if (Array.isArray(h))
                    for (var m = 0, g = h.length; m < g; m++) f(h[m]);
                else "string" === typeof h ? -1 !== h.indexOf(",") ? f(h.split(",")) : (h = e.inArray(h.trim(), d), -1 !== h && c.push(b[h].inst)) : "number" === typeof h && c.push(b[h].inst)
            };
        f(a);
        return c
    };
    t.buttonSelector = function(a, b) {
        for (var c = [], d = function(g, n, k) {
                for (var p, l, v = 0, u = n.length; v < u; v++)
                    if (p = n[v]) l = k !== r ? k + v :
                        v + "", g.push({ node: p.node, name: p.conf.name, idx: l }), p.buttons && d(g, p.buttons, l + "-")
            }, f = function(g, n) {
                var k, p = [];
                d(p, n.s.buttons);
                var l = e.map(p, function(v) { return v.node });
                if (Array.isArray(g) || g instanceof e)
                    for (l = 0, k = g.length; l < k; l++) f(g[l], n);
                else if (null === g || g === r || "*" === g)
                    for (l = 0, k = p.length; l < k; l++) c.push({ inst: n, node: p[l].node });
                else if ("number" === typeof g) c.push({ inst: n, node: n.s.buttons[g].node });
                else if ("string" === typeof g)
                    if (-1 !== g.indexOf(","))
                        for (p = g.split(","), l = 0, k = p.length; l < k; l++) f(p[l].trim(),
                            n);
                    else if (g.match(/^\d+(\-\d+)*$/)) l = e.map(p, function(v) { return v.idx }), c.push({ inst: n, node: p[e.inArray(g, l)].node });
                else if (-1 !== g.indexOf(":name"))
                    for (g = g.replace(":name", ""), l = 0, k = p.length; l < k; l++) p[l].name === g && c.push({ inst: n, node: p[l].node });
                else e(l).filter(g).each(function() { c.push({ inst: n, node: this }) });
                else "object" === typeof g && g.nodeName && (p = e.inArray(g, l), -1 !== p && c.push({ inst: n, node: l[p] }))
            }, h = 0, m = a.length; h < m; h++) f(b, a[h]);
        return c
    };
    t.stripData = function(a, b) {
        if ("string" !== typeof a) return a;
        a = a.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, "");
        a = a.replace(/<!\-\-.*?\-\->/g, "");
        if (!b || b.stripHtml) a = a.replace(/<[^>]*>/g, "");
        if (!b || b.trim) a = a.replace(/^\s+|\s+$/g, "");
        if (!b || b.stripNewlines) a = a.replace(/\n/g, " ");
        if (!b || b.decodeEntities) G.innerHTML = a, a = G.value;
        return a
    };
    t.defaults = {
        buttons: ["copy", "excel", "csv", "pdf", "print"],
        name: "main",
        tabIndex: 0,
        dom: {
            container: { tag: "div", className: "dt-buttons" },
            collection: { tag: "div", className: "" },
            button: {
                tag: "button",
                className: "dt-button",
                active: "active",
                disabled: "disabled"
            },
            buttonLiner: { tag: "span", className: "" }
        }
    };
    t.version = "1.7.1";
    e.extend(x, {
        collection: {
            text: function(a) { return a.i18n("buttons.collection", "Collection") },
            className: "buttons-collection",
            init: function(a, b, c) { b.attr("aria-expanded", !1) },
            action: function(a, b, c, d) {
                a.stopPropagation();
                d._collection.parents("body").length ? this.popover(!1, d) : this.popover(d._collection, d)
            },
            attr: { "aria-haspopup": !0 }
        },
        copy: function(a, b) { if (x.copyHtml5) return "copyHtml5" },
        csv: function(a, b) {
            if (x.csvHtml5 &&
                x.csvHtml5.available(a, b)) return "csvHtml5"
        },
        excel: function(a, b) { if (x.excelHtml5 && x.excelHtml5.available(a, b)) return "excelHtml5" },
        pdf: function(a, b) { if (x.pdfHtml5 && x.pdfHtml5.available(a, b)) return "pdfHtml5" },
        pageLength: function(a) {
            a = a.settings()[0].aLengthMenu;
            var b = [],
                c = [];
            if (Array.isArray(a[0])) b = a[0], c = a[1];
            else
                for (var d = 0; d < a.length; d++) {
                    var f = a[d];
                    e.isPlainObject(f) ? (b.push(f.value), c.push(f.label)) : (b.push(f), c.push(f))
                }
            return {
                extend: "collection",
                text: function(h) {
                    return h.i18n("buttons.pageLength", { "-1": "Show all rows", _: "Show %d rows" }, h.page.len())
                },
                className: "buttons-page-length",
                autoClose: !0,
                buttons: e.map(b, function(h, m) {
                    return {
                        text: c[m],
                        className: "button-page-length",
                        action: function(g, n) { n.page.len(h).draw() },
                        init: function(g, n, k) {
                            var p = this;
                            n = function() { p.active(g.page.len() === h) };
                            g.on("length.dt" + k.namespace, n);
                            n()
                        },
                        destroy: function(g, n, k) { g.off("length.dt" + k.namespace) }
                    }
                }),
                init: function(h, m, g) {
                    var n = this;
                    h.on("length.dt" + g.namespace, function() { n.text(g.text) })
                },
                destroy: function(h, m,
                    g) { h.off("length.dt" + g.namespace) }
            }
        }
    });
    q.Api.register("buttons()", function(a, b) {
        b === r && (b = a, a = r);
        this.selector.buttonGroup = a;
        var c = this.iterator(!0, "table", function(d) { if (d._buttons) return t.buttonSelector(t.instanceSelector(a, d._buttons), b) }, !0);
        c._groupSelector = a;
        return c
    });
    q.Api.register("button()", function(a, b) {
        a = this.buttons(a, b);
        1 < a.length && a.splice(1, a.length);
        return a
    });
    q.Api.registerPlural("buttons().active()", "button().active()", function(a) {
        return a === r ? this.map(function(b) { return b.inst.active(b.node) }) :
            this.each(function(b) { b.inst.active(b.node, a) })
    });
    q.Api.registerPlural("buttons().action()", "button().action()", function(a) { return a === r ? this.map(function(b) { return b.inst.action(b.node) }) : this.each(function(b) { b.inst.action(b.node, a) }) });
    q.Api.register(["buttons().enable()", "button().enable()"], function(a) { return this.each(function(b) { b.inst.enable(b.node, a) }) });
    q.Api.register(["buttons().disable()", "button().disable()"], function() { return this.each(function(a) { a.inst.disable(a.node) }) });
    q.Api.registerPlural("buttons().nodes()",
        "button().node()",
        function() {
            var a = e();
            e(this.each(function(b) { a = a.add(b.inst.node(b.node)) }));
            return a
        });
    q.Api.registerPlural("buttons().processing()", "button().processing()", function(a) { return a === r ? this.map(function(b) { return b.inst.processing(b.node) }) : this.each(function(b) { b.inst.processing(b.node, a) }) });
    q.Api.registerPlural("buttons().text()", "button().text()", function(a) { return a === r ? this.map(function(b) { return b.inst.text(b.node) }) : this.each(function(b) { b.inst.text(b.node, a) }) });
    q.Api.registerPlural("buttons().trigger()",
        "button().trigger()",
        function() { return this.each(function(a) { a.inst.node(a.node).trigger("click") }) });
    q.Api.register("button().popover()", function(a, b) { return this.map(function(c) { return c.inst._popover(a, this.button(this[0].node), b) }) });
    q.Api.register("buttons().containers()", function() {
        var a = e(),
            b = this._groupSelector;
        this.iterator(!0, "table", function(c) { if (c._buttons) { c = t.instanceSelector(b, c._buttons); for (var d = 0, f = c.length; d < f; d++) a = a.add(c[d].container()) } });
        return a
    });
    q.Api.register("buttons().container()",
        function() { return this.containers().eq(0) });
    q.Api.register("button().add()", function(a, b) {
        var c = this.context;
        c.length && (c = t.instanceSelector(this._groupSelector, c[0]._buttons), c.length && c[0].add(b, a));
        return this.button(this._groupSelector, a)
    });
    q.Api.register("buttons().destroy()", function() { this.pluck("inst").unique().each(function(a) { a.destroy() }); return this });
    q.Api.registerPlural("buttons().remove()", "buttons().remove()", function() { this.each(function(a) { a.inst.remove(a.node) }); return this });
    var A;
    q.Api.register("buttons.info()", function(a, b, c) {
        var d = this;
        if (!1 === a) return this.off("destroy.btn-info"), C(e("#datatables_buttons_info"), 400, function() { e(this).remove() }), clearTimeout(A), A = null, this;
        A && clearTimeout(A);
        e("#datatables_buttons_info").length && e("#datatables_buttons_info").remove();
        a = a ? "<h2>" + a + "</h2>" : "";
        B(e('<div id="datatables_buttons_info" class="dt-button-info"/>').html(a).append(e("<div/>")["string" === typeof b ? "html" : "append"](b)).css("display", "none").appendTo("body"));
        c !== r && 0 !==
            c && (A = setTimeout(function() { d.buttons.info(!1) }, c));
        this.on("destroy.btn-info", function() { d.buttons.info(!1) });
        return this
    });
    q.Api.register("buttons.exportData()", function(a) { if (this.context.length) return L(new q.Api(this.context[0]), a) });
    q.Api.register("buttons.exportInfo()", function(a) {
        a || (a = {});
        var b = a;
        var c = "*" === b.filename && "*" !== b.title && b.title !== r && null !== b.title && "" !== b.title ? b.title : b.filename;
        "function" === typeof c && (c = c());
        c === r || null === c ? c = null : (-1 !== c.indexOf("*") && (c = c.replace("*", e("head > title").text()).trim()),
            c = c.replace(/[^a-zA-Z0-9_\u00A1-\uFFFF\.,\-_ !\(\)]/g, ""), (b = D(b.extension)) || (b = ""), c += b);
        b = D(a.title);
        b = null === b ? null : -1 !== b.indexOf("*") ? b.replace("*", e("head > title").text() || "Exported data") : b;
        return { filename: c, title: b, messageTop: H(this, a.message || a.messageTop, "top"), messageBottom: H(this, a.messageBottom, "bottom") }
    });
    var D = function(a) { return null === a || a === r ? null : "function" === typeof a ? a() : a },
        H = function(a, b, c) {
            b = D(b);
            if (null === b) return null;
            a = e("caption", a.table().container()).eq(0);
            return "*" ===
                b ? a.css("caption-side") !== c ? null : a.length ? a.text() : "" : b
        },
        G = e("<textarea/>")[0],
        L = function(a, b) {
            var c = e.extend(!0, {}, { rows: null, columns: "", modifier: { search: "applied", order: "applied" }, orthogonal: "display", stripHtml: !0, stripNewlines: !0, decodeEntities: !0, trim: !0, format: { header: function(u) { return t.stripData(u, c) }, footer: function(u) { return t.stripData(u, c) }, body: function(u) { return t.stripData(u, c) } }, customizeData: null }, b);
            b = a.columns(c.columns).indexes().map(function(u) {
                var z = a.column(u).header();
                return c.format.header(z.innerHTML,
                    u, z)
            }).toArray();
            var d = a.table().footer() ? a.columns(c.columns).indexes().map(function(u) { var z = a.column(u).footer(); return c.format.footer(z ? z.innerHTML : "", u, z) }).toArray() : null,
                f = e.extend({}, c.modifier);
            a.select && "function" === typeof a.select.info && f.selected === r && a.rows(c.rows, e.extend({ selected: !0 }, f)).any() && e.extend(f, { selected: !0 });
            f = a.rows(c.rows, f).indexes().toArray();
            var h = a.cells(f, c.columns);
            f = h.render(c.orthogonal).toArray();
            h = h.nodes().toArray();
            for (var m = b.length, g = [], n = 0, k = 0, p = 0 < m ? f.length /
                    m : 0; k < p; k++) {
                for (var l = [m], v = 0; v < m; v++) l[v] = c.format.body(f[n], k, v, h[n]), n++;
                g[k] = l
            }
            b = { header: b, footer: d, body: g };
            c.customizeData && c.customizeData(b);
            return b
        };
    e.fn.dataTable.Buttons = t;
    e.fn.DataTable.Buttons = t;
    e(w).on("init.dt plugin-init.dt", function(a, b) { "dt" === a.namespace && (a = b.oInit.buttons || q.defaults.buttons) && !b._buttons && (new t(b, a)).container() });
    q.ext.feature.push({ fnInit: E, cFeature: "B" });
    q.ext.features && q.ext.features.register("buttons", E);
    return t
});