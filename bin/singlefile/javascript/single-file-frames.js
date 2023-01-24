!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?t(exports):"function"==typeof define&&define.amd?define(["exports"],t):t((e="undefined"!=typeof globalThis?globalThis:e||self).singlefile={})}(this,(function(e){"use strict";const t="single-file-load-image",s="single-file-image-loaded",o=(e,t,s)=>globalThis.addEventListener(e,t,s),n=e=>{try{globalThis.dispatchEvent(e)}catch(e){}},i=globalThis.CustomEvent,a=globalThis.document,r=globalThis.Document;let l;l=window._singleFile_fontFaces?window._singleFile_fontFaces:window._singleFile_fontFaces=new Map,a instanceof r&&(o("single-file-new-font-face",(e=>{const t=e.detail,s=Object.assign({},t);delete s.src,l.set(JSON.stringify(s),t)})),o("single-file-delete-font",(e=>{const t=e.detail,s=Object.assign({},t);delete s.src,l.delete(JSON.stringify(s))})),o("single-file-clear-fonts",(()=>l=new Map)));const d=new RegExp("\\\\([\\da-f]{1,6}[\\x20\\t\\r\\n\\f]?|([\\x20\\t\\r\\n\\f])|.)","ig");const c="data-single-file-removed-content",m="data-single-file-hidden-content",u="data-single-file-kept-content",g="data-single-file-hidden-frame",f="data-single-file-preserved-space-element",h="data-single-file-shadow-root-element",p="data-single-file-image",b="data-single-file-poster",y="data-single-file-video",E="data-single-file-canvas",w="data-single-file-movable-style",T="data-single-file-input-value",I="data-single-file-lazy-loaded-src",v="data-single-file-stylesheet",A="data-single-file-disabled-noscript",S="data-single-file-async-script",R="*:not(base):not(link):not(meta):not(noscript):not(script):not(style):not(template):not(title)",N=["NOSCRIPT","DISABLED-NOSCRIPT","META","LINK","STYLE","TITLE","TEMPLATE","SOURCE","OBJECT","SCRIPT","HEAD","BODY"],_=/^'(.*?)'$/,F=/^"(.*?)"$/,M={regular:"400",normal:"400",bold:"700",bolder:"700",lighter:"100"},C="single-file-ui-element",q="data:,";function x(e,t,s,o,n={usedFonts:new Map,canvases:[],images:[],posters:[],videos:[],shadowRoots:[],markedElements:[]},i){return Array.from(s.childNodes).filter((t=>t instanceof e.HTMLElement||t instanceof e.SVGElement)).forEach((s=>{let a,r,l;if(!o.autoSaveExternalSave&&(o.removeHiddenElements||o.removeUnusedFonts||o.compressHTML)&&(l=e.getComputedStyle(s),s instanceof e.HTMLElement&&o.removeHiddenElements&&(r=(i||s.closest("html > head"))&&N.includes(s.tagName)||s.closest("details"),r||(a=i||O(s,l),a&&(s.setAttribute(m,""),n.markedElements.push(s)))),!a)){if(o.compressHTML&&l){const e=l.getPropertyValue("white-space");e&&e.startsWith("pre")&&(s.setAttribute(f,""),n.markedElements.push(s))}o.removeUnusedFonts&&(P(l,o,n.usedFonts),P(e.getComputedStyle(s,":first-letter"),o,n.usedFonts),P(e.getComputedStyle(s,":before"),o,n.usedFonts),P(e.getComputedStyle(s,":after"),o,n.usedFonts))}!function(e,t,s,o,n,i,a){if("CANVAS"==s.tagName)try{n.canvases.push({dataURI:s.toDataURL("image/png","")}),s.setAttribute(E,n.canvases.length-1),n.markedElements.push(s)}catch(e){}if("IMG"==s.tagName){const t={currentSrc:i?q:o.loadDeferredImages&&s.getAttribute(I)||s.currentSrc};n.images.push(t),s.setAttribute(p,n.images.length-1),n.markedElements.push(s),s.removeAttribute(I);try{a=a||e.getComputedStyle(s)}catch(e){}if(a){t.size=function(e,t,s){let o=t.naturalWidth,n=t.naturalHeight;if(!o&&!n){const i=null==t.getAttribute("style");let a,r,l,d,c,m,u,g,f=!1;if("content-box"==(s=s||e.getComputedStyle(t)).getPropertyValue("box-sizing")){const e=t.style.getPropertyValue("box-sizing"),s=t.style.getPropertyPriority("box-sizing"),o=t.clientWidth;t.style.setProperty("box-sizing","border-box","important"),f=t.clientWidth!=o,e?t.style.setProperty("box-sizing",e,s):t.style.removeProperty("box-sizing")}a=U("padding-left",s),r=U("padding-right",s),l=U("padding-top",s),d=U("padding-bottom",s),f?(c=U("border-left-width",s),m=U("border-right-width",s),u=U("border-top-width",s),g=U("border-bottom-width",s)):c=m=u=g=0,o=Math.max(0,t.clientWidth-a-r-c-m),n=Math.max(0,t.clientHeight-l-d-u-g),i&&t.removeAttribute("style")}return{pxWidth:o,pxHeight:n}}(e,s,a);const o=a.getPropertyValue("box-shadow"),n=a.getPropertyValue("background-image");o&&"none"!=o||n&&"none"!=n||!(t.size.pxWidth>1||t.size.pxHeight>1)||(t.replaceable=!0,t.backgroundColor=a.getPropertyValue("background-color"),t.objectFit=a.getPropertyValue("object-fit"),t.boxSizing=a.getPropertyValue("box-sizing"),t.objectPosition=a.getPropertyValue("object-position"))}}if("VIDEO"==s.tagName){const o=s.currentSrc;if(o&&!o.startsWith("blob:")&&!o.startsWith("data:")){const t=e.getComputedStyle(s.parentNode).getPropertyValue("position");n.videos.push({positionParent:t,src:o,size:{pxWidth:s.clientWidth,pxHeight:s.clientHeight},currentTime:s.currentTime}),s.setAttribute(y,n.videos.length-1)}if(!s.poster){const e=t.createElement("canvas"),o=e.getContext("2d");e.width=s.clientWidth,e.height=s.clientHeight;try{o.drawImage(s,0,0,e.width,e.height),n.posters.push(e.toDataURL("image/png","")),s.setAttribute(b,n.posters.length-1),n.markedElements.push(s)}catch(e){}}}"IFRAME"==s.tagName&&i&&o.removeHiddenElements&&(s.setAttribute(g,""),n.markedElements.push(s));"INPUT"==s.tagName&&("password"!=s.type&&(s.setAttribute(T,s.value),n.markedElements.push(s)),"radio"!=s.type&&"checkbox"!=s.type||(s.setAttribute(T,s.checked),n.markedElements.push(s)));"TEXTAREA"==s.tagName&&(s.setAttribute(T,s.value),n.markedElements.push(s));"SELECT"==s.tagName&&s.querySelectorAll("option").forEach((e=>{e.selected&&(e.setAttribute(T,""),n.markedElements.push(e))}));"SCRIPT"==s.tagName&&(s.async&&""!=s.getAttribute("async")&&"async"!=s.getAttribute("async")&&(s.setAttribute(S,""),n.markedElements.push(s)),s.textContent=s.textContent.replace(/<\/script>/gi,"<\\/script>"))}(e,t,s,o,n,a,l);const d=!(s instanceof e.SVGElement)&&k(s);if(d&&!s.classList.contains(C)){const i={};s.setAttribute(h,n.shadowRoots.length),n.markedElements.push(s),n.shadowRoots.push(i),x(e,t,d,o,n,a),i.content=d.innerHTML,i.mode=d.mode;try{d.adoptedStyleSheets&&d.adoptedStyleSheets.length&&(i.adoptedStyleSheets=Array.from(d.adoptedStyleSheets).map((e=>Array.from(e.cssRules).map((e=>e.cssText)).join("\n"))))}catch(e){}}x(e,t,s,o,n,a),!o.autoSaveExternalSave&&o.removeHiddenElements&&i&&(r||""==s.getAttribute(u)?s.parentElement&&(s.parentElement.setAttribute(u,""),n.markedElements.push(s.parentElement)):a&&(s.setAttribute(c,""),n.markedElements.push(s)))})),n}function P(e,t,s){if(e){const o=e.getPropertyValue("font-style")||"normal";e.getPropertyValue("font-family").split(",").forEach((n=>{if(n=D(n),!t.loadedFonts||t.loadedFonts.find((e=>D(e.family)==n&&e.style==o))){const t=(i=e.getPropertyValue("font-weight"),M[i.toLowerCase().trim()]||i),a=e.getPropertyValue("font-variant")||"normal",r=[n,t,o,a];s.set(JSON.stringify(r),[n,t,o,a])}var i}))}}function k(e){const t=globalThis.chrome;if(e.openOrClosedShadowRoot)return e.openOrClosedShadowRoot;if(!(t&&t.dom&&t.dom.openOrClosedShadowRoot))return e.shadowRoot;try{return t.dom.openOrClosedShadowRoot(e)}catch(t){return e.shadowRoot}}function D(e=""){return function(e){e=e.match(_)?e.replace(_,"$1"):e.replace(F,"$1");return e.trim()}((t=e.trim(),t.replace(d,((e,t,s)=>{const o="0x"+t-65536;return o!=o||s?t:o<0?String.fromCharCode(o+65536):String.fromCharCode(o>>10|55296,1023&o|56320)})))).toLowerCase();var t}function O(e,t){let s=!1;if(t){const o=t.getPropertyValue("display"),n=t.getPropertyValue("opacity"),i=t.getPropertyValue("visibility");if(s="none"==o,!s&&("0"==n||"hidden"==i)&&e.getBoundingClientRect){const t=e.getBoundingClientRect();s=!t.width&&!t.height}}return Boolean(s)}function L(e){if(e){const t=[];return e.querySelectorAll("style").forEach(((s,o)=>{try{const n=e.createElement("style");n.textContent=s.textContent,e.body.appendChild(n);const i=n.sheet;n.remove(),i&&i.cssRules.length==s.sheet.cssRules.length||(s.setAttribute(v,o),t[o]=Array.from(s.sheet.cssRules).map((e=>e.cssText)).join("\n"))}catch(e){}})),t}}function U(e,t){if(t.getPropertyValue(e).endsWith("px"))return parseFloat(t.getPropertyValue(e))}const V=I,W=C,z="attributes",H=globalThis.browser,B=globalThis.document,j=globalThis.MutationObserver,J=(e,t,s)=>globalThis.addEventListener(e,t,s),Y=(e,t,s)=>globalThis.removeEventListener(e,t,s),G=new Map;let $;async function K(e){if(B.documentElement){G.clear();const o=Math.max(B.documentElement.scrollHeight-1.5*B.documentElement.clientHeight,0),a=Math.max(B.documentElement.scrollWidth-1.5*B.documentElement.clientWidth,0);if(globalThis.scrollY<=o&&globalThis.scrollX<=a)return function(e){return $=0,new Promise((async o=>{let a;const r=new Set,l=new j((async t=>{if((t=t.filter((e=>e.type==z))).length){t.filter((e=>{if("src"==e.attributeName&&(e.target.setAttribute(V,e.target.src),e.target.addEventListener("load",c)),"src"==e.attributeName||"srcset"==e.attributeName||"SOURCE"==e.target.tagName)return!e.target.classList||!e.target.classList.contains(W)})).length&&(a=!0,await Z(l,e,g),r.size||await X(l,e,g))}}));async function d(t){await ee("idleTimeout",(async()=>{a?$<10&&($++,se("idleTimeout"),await d(Math.max(500,t/2))):(se("loadTimeout"),se("maxTimeout"),Q(l,e,g))}),t,e.loadDeferredImagesNativeTimeout)}function c(e){const t=e.target;t.removeAttribute(V),t.removeEventListener("load",c)}async function m(t){a=!0,await Z(l,e,g),await X(l,e,g),t.detail&&r.add(t.detail)}async function u(t){await Z(l,e,g),await X(l,e,g),r.delete(t.detail),r.size||await X(l,e,g)}function g(e){l.disconnect(),Y(t,m),Y(s,u),o(e)}await d(2*e.loadDeferredImagesMaxIdleTime),await Z(l,e,g),l.observe(B,{subtree:!0,childList:!0,attributes:!0}),J(t,m),J(s,u),function(e){e.loadDeferredImagesBlockCookies&&n(new i("single-file-block-cookies-start")),e.loadDeferredImagesBlockStorage&&n(new i("single-file-block-storage-start")),e.loadDeferredImagesDispatchScrollEvent&&n(new i("single-file-dispatch-scroll-event-start")),e.loadDeferredImagesKeepZoomLevel?n(new i("single-file-load-deferred-images-keep-zoom-level-start")):n(new i("single-file-load-deferred-images-start"))}(e)}))}(e)}}async function X(e,t,s){await ee("loadTimeout",(()=>Q(e,t,s)),t.loadDeferredImagesMaxIdleTime,t.loadDeferredImagesNativeTimeout)}async function Z(e,t,s){await ee("maxTimeout",(async()=>{await se("loadTimeout"),await Q(e,t,s)}),10*t.loadDeferredImagesMaxIdleTime,t.loadDeferredImagesNativeTimeout)}async function Q(e,t,s){await se("idleTimeout"),function(e){e.loadDeferredImagesBlockCookies&&n(new i("single-file-block-cookies-end")),e.loadDeferredImagesBlockStorage&&n(new i("single-file-block-storage-end")),e.loadDeferredImagesDispatchScrollEvent&&n(new i("single-file-dispatch-scroll-event-end")),e.loadDeferredImagesKeepZoomLevel?n(new i("single-file-load-deferred-images-keep-zoom-level-end")):n(new i("single-file-load-deferred-images-end"))}(t),await ee("endTimeout",(async()=>{await se("maxTimeout"),s()}),t.loadDeferredImagesMaxIdleTime/2,t.loadDeferredImagesNativeTimeout),e.disconnect()}async function ee(e,t,s,o){if(H&&H.runtime&&H.runtime.sendMessage&&!o){if(!G.get(e)||!G.get(e).pending){const o={callback:t,pending:!0};G.set(e,o);try{await H.runtime.sendMessage({method:"singlefile.lazyTimeout.setTimeout",type:e,delay:s})}catch(o){te(e,t,s)}o.pending=!1}}else te(e,t,s)}function te(e,t,s){const o=G.get(e);o&&globalThis.clearTimeout(o),G.set(e,t),globalThis.setTimeout(t,s)}async function se(e){if(H&&H.runtime&&H.runtime.sendMessage)try{await H.runtime.sendMessage({method:"singlefile.lazyTimeout.clearTimeout",type:e})}catch(t){oe(e)}else oe(e)}function oe(e){const t=G.get(e);G.delete(e),t&&globalThis.clearTimeout(t)}H&&H.runtime&&H.runtime.onMessage&&H.runtime.onMessage.addListener&&H.runtime.onMessage.addListener((e=>{if("singlefile.lazyTimeout.onTimeout"==e.method){const t=G.get(e.type);if(t){G.delete(e.type);try{t.callback()}catch(t){oe(e.type)}}}}));const ne={ON_BEFORE_CAPTURE_EVENT_NAME:"single-file-on-before-capture",ON_AFTER_CAPTURE_EVENT_NAME:"single-file-on-after-capture",WIN_ID_ATTRIBUTE_NAME:"data-single-file-win-id",preProcessDoc:function(e,t,s){e.querySelectorAll("noscript:not([data-single-file-disabled-noscript])").forEach((e=>{e.setAttribute(A,e.textContent),e.textContent=""})),function(e){e.querySelectorAll("meta[http-equiv=refresh]").forEach((e=>{e.removeAttribute("http-equiv"),e.setAttribute("disabled-http-equiv","refresh")}))}(e),e.head&&e.head.querySelectorAll(R).forEach((e=>e.hidden=!0)),e.querySelectorAll("svg foreignObject").forEach((e=>{const t=e.querySelectorAll("html > head > "+R+", html > body > "+R);t.length&&(Array.from(e.childNodes).forEach((e=>e.remove())),t.forEach((t=>e.appendChild(t))))}));const o=new Map;let n;return t&&e.documentElement?(e.querySelectorAll("button button, a a, p div").forEach((t=>{const s=e.createElement("template");s.setAttribute("data-single-file-invalid-element",""),s.content.appendChild(t.cloneNode(!0)),o.set(t,s),t.replaceWith(s)})),n=x(t,e,e.documentElement,s),s.moveStylesInHead&&e.querySelectorAll("body style, body ~ style").forEach((e=>{const s=t.getComputedStyle(e);s&&O(e,s)&&(e.setAttribute(w,""),n.markedElements.push(e))}))):n={canvases:[],images:[],posters:[],videos:[],usedFonts:[],shadowRoots:[],markedElements:[]},{canvases:n.canvases,fonts:Array.from(l.values()),stylesheets:L(e),images:n.images,posters:n.posters,videos:n.videos,usedFonts:Array.from(n.usedFonts.values()),shadowRoots:n.shadowRoots,referrer:e.referrer,markedElements:n.markedElements,invalidElements:o}},serialize:function(e){const t=e.doctype;let s="";return t&&(s="<!DOCTYPE "+t.nodeName,t.publicId?(s+=' PUBLIC "'+t.publicId+'"',t.systemId&&(s+=' "'+t.systemId+'"')):t.systemId&&(s+=' SYSTEM "'+t.systemId+'"'),t.internalSubset&&(s+=" ["+t.internalSubset+"]"),s+="> "),s+e.documentElement.outerHTML},postProcessDoc:function(e,t,s){if(e.querySelectorAll("[data-single-file-disabled-noscript]").forEach((e=>{e.textContent=e.getAttribute(A),e.removeAttribute(A)})),e.querySelectorAll("meta[disabled-http-equiv]").forEach((e=>{e.setAttribute("http-equiv",e.getAttribute("disabled-http-equiv")),e.removeAttribute("disabled-http-equiv")})),e.head&&e.head.querySelectorAll("*:not(base):not(link):not(meta):not(noscript):not(script):not(style):not(template):not(title)").forEach((e=>e.removeAttribute("hidden"))),!t){const s=[c,g,m,f,p,b,y,E,T,h,v,S];t=e.querySelectorAll(s.map((e=>"["+e+"]")).join(","))}t.forEach((e=>{e.removeAttribute(c),e.removeAttribute(m),e.removeAttribute(u),e.removeAttribute(g),e.removeAttribute(f),e.removeAttribute(p),e.removeAttribute(b),e.removeAttribute(y),e.removeAttribute(E),e.removeAttribute(T),e.removeAttribute(h),e.removeAttribute(v),e.removeAttribute(S),e.removeAttribute(w)})),s&&Array.from(s.entries()).forEach((([e,t])=>t.replaceWith(e)))},getShadowRoot:k},ie="__frameTree__::",ae='iframe, frame, object[type="text/html"][data]',re="singlefile.frameTree.initRequest",le="singlefile.frameTree.ackInitRequest",de="singlefile.frameTree.cleanupRequest",ce="singlefile.frameTree.initResponse",me=5e3,ue=".",ge=globalThis.window==globalThis.top,fe=globalThis.browser,he=globalThis.top,pe=globalThis.MessageChannel,be=globalThis.document;let ye,Ee=globalThis.sessions;var we,Te,Ie;function ve(){return globalThis.crypto.getRandomValues(new Uint32Array(32)).join("")}async function Ae(e){const t=e.sessionId,s=globalThis._singleFile_waitForUserScript;delete globalThis._singleFile_cleaningUp,ge||(ye=globalThis.frameId=e.windowId),Ne(be,e.options,ye,t),ge||(e.options.userScriptEnabled&&s&&await s(ne.ON_BEFORE_CAPTURE_EVENT_NAME),Ce({frames:[xe(be,globalThis,ye,e.options)],sessionId:t,requestedFrameId:be.documentElement.dataset.requestedFrameId&&ye}),e.options.userScriptEnabled&&s&&await s(ne.ON_AFTER_CAPTURE_EVENT_NAME),delete be.documentElement.dataset.requestedFrameId)}function Se(e){if(!globalThis._singleFile_cleaningUp){globalThis._singleFile_cleaningUp=!0;const t=e.sessionId;Me(Pe(be),e.windowId,t)}}function Re(e){e.frames.forEach((t=>_e("responseTimeouts",e.sessionId,t.windowId)));const t=Ee.get(e.sessionId);if(t){e.requestedFrameId&&(t.requestedFrameId=e.requestedFrameId),e.frames.forEach((e=>{let s=t.frames.find((t=>e.windowId==t.windowId));s||(s={windowId:e.windowId},t.frames.push(s)),s.processed||(s.content=e.content,s.baseURI=e.baseURI,s.title=e.title,s.canvases=e.canvases,s.fonts=e.fonts,s.stylesheets=e.stylesheets,s.images=e.images,s.posters=e.posters,s.videos=e.videos,s.usedFonts=e.usedFonts,s.shadowRoots=e.shadowRoots,s.processed=e.processed)}));t.frames.filter((e=>!e.processed)).length||(t.frames=t.frames.sort(((e,t)=>t.windowId.split(ue).length-e.windowId.split(ue).length)),t.resolve&&(t.requestedFrameId&&t.frames.forEach((e=>{e.windowId==t.requestedFrameId&&(e.requestedFrame=!0)})),t.resolve(t.frames)))}}function Ne(e,t,s,o){const n=Pe(e);!function(e,t,s,o,n){const i=[];let a;Ee.get(n)?a=Ee.get(n).requestTimeouts:(a={},Ee.set(n,{requestTimeouts:a}));t.forEach(((e,t)=>{const s=o+ue+t;e.setAttribute(ne.WIN_ID_ATTRIBUTE_NAME,s),i.push({windowId:s})})),Ce({frames:i,sessionId:n,requestedFrameId:e.documentElement.dataset.requestedFrameId&&o}),t.forEach(((e,t)=>{const i=o+ue+t;try{qe(e.contentWindow,{method:re,windowId:i,sessionId:n,options:s})}catch(e){}a[i]=globalThis.setTimeout((()=>Ce({frames:[{windowId:i,processed:!0}],sessionId:n})),me)})),delete e.documentElement.dataset.requestedFrameId}(e,n,t,s,o),n.length&&function(e,t,s,o,n){const i=[];t.forEach(((e,t)=>{const a=o+ue+t;let r;try{r=e.contentDocument}catch(e){}if(r)try{const t=e.contentWindow;t.stop(),_e("requestTimeouts",n,a),Ne(r,s,a,n),i.push(xe(r,t,a,s))}catch(e){i.push({windowId:a,processed:!0})}})),Ce({frames:i,sessionId:n,requestedFrameId:e.documentElement.dataset.requestedFrameId&&o}),delete e.documentElement.dataset.requestedFrameId}(e,n,t,s,o)}function _e(e,t,s){const o=Ee.get(t);if(o&&o[e]){const t=o[e][s];t&&(globalThis.clearTimeout(t),delete o[e][s])}}function Fe(e,t){const s=Ee.get(e);s&&s.responseTimeouts&&(s.responseTimeouts[t]=globalThis.setTimeout((()=>Ce({frames:[{windowId:t,processed:!0}],sessionId:e})),1e4))}function Me(e,t,s){e.forEach(((e,o)=>{const n=t+ue+o;e.removeAttribute(ne.WIN_ID_ATTRIBUTE_NAME);try{qe(e.contentWindow,{method:de,windowId:n,sessionId:s})}catch(e){}})),e.forEach(((e,o)=>{const n=t+ue+o;let i;try{i=e.contentDocument}catch(e){}if(i)try{Me(Pe(i),n,s)}catch(e){}}))}function Ce(e){e.method=ce;try{he.singlefile.processors.frameTree.initResponse(e)}catch(t){qe(he,e,!0)}}function qe(e,t,s){if(e==he&&fe&&fe.runtime&&fe.runtime.sendMessage)fe.runtime.sendMessage(t);else if(s){const s=new pe;e.postMessage(ie+JSON.stringify({method:t.method,sessionId:t.sessionId}),"*",[s.port2]),s.port1.postMessage(t)}else e.postMessage(ie+JSON.stringify(t),"*")}function xe(e,t,s,o){const n=ne.preProcessDoc(e,t,o),i=ne.serialize(e);ne.postProcessDoc(e,n.markedElements,n.invalidElements);return{windowId:s,content:i,baseURI:e.baseURI.split("#")[0],title:e.title,canvases:n.canvases,fonts:n.fonts,stylesheets:n.stylesheets,images:n.images,posters:n.posters,videos:n.videos,usedFonts:n.usedFonts,shadowRoots:n.shadowRoots,processed:!0}}function Pe(e){let t=Array.from(e.querySelectorAll(ae));return e.querySelectorAll("*").forEach((e=>{const s=ne.getShadowRoot(e);s&&(t=t.concat(...s.querySelectorAll(ae)))})),t}Ee||(Ee=globalThis.sessions=new Map),ge&&(ye="0",fe&&fe.runtime&&fe.runtime.onMessage&&fe.runtime.onMessage.addListener&&fe.runtime.onMessage.addListener((e=>e.method==ce?(Re(e),Promise.resolve({})):e.method==le?(_e("requestTimeouts",e.sessionId,e.windowId),Fe(e.sessionId,e.windowId),Promise.resolve({})):void 0))),we="message",Te=async e=>{if("string"==typeof e.data&&e.data.startsWith(ie)){e.preventDefault(),e.stopPropagation();const t=JSON.parse(e.data.substring(ie.length));t.method==re?(e.source&&qe(e.source,{method:le,windowId:t.windowId,sessionId:t.sessionId}),ge||(globalThis.stop(),t.options.loadDeferredImages&&K(t.options),await Ae(t))):t.method==le?(_e("requestTimeouts",t.sessionId,t.windowId),Fe(t.sessionId,t.windowId)):t.method==de?Se(t):t.method==ce&&Ee.get(t.sessionId)&&(e.ports[0].onmessage=e=>Re(e.data))}},Ie=!0,globalThis.addEventListener(we,Te,Ie),e.TIMEOUT_INIT_REQUEST_MESSAGE=me,e.cleanup=function(e){Ee.delete(e),Se({windowId:ye,sessionId:e,options:{sessionId:e}})},e.getAsync=function(e){const t=ve();return e=JSON.parse(JSON.stringify(e)),new Promise((s=>{Ee.set(t,{frames:[],requestTimeouts:{},responseTimeouts:{},resolve:e=>{e.sessionId=t,s(e)}}),Ae({windowId:ye,sessionId:t,options:e})}))},e.getSync=function(e){const t=ve();e=JSON.parse(JSON.stringify(e)),Ee.set(t,{frames:[],requestTimeouts:{},responseTimeouts:{}}),function(e){const t=e.sessionId,s=globalThis._singleFile_waitForUserScript;delete globalThis._singleFile_cleaningUp,ge||(ye=globalThis.frameId=e.windowId);Ne(be,e.options,ye,t),ge||(e.options.userScriptEnabled&&s&&s(ne.ON_BEFORE_CAPTURE_EVENT_NAME),Ce({frames:[xe(be,globalThis,ye,e.options)],sessionId:t,requestedFrameId:be.documentElement.dataset.requestedFrameId&&ye}),e.options.userScriptEnabled&&s&&s(ne.ON_AFTER_CAPTURE_EVENT_NAME),delete be.documentElement.dataset.requestedFrameId)}({windowId:ye,sessionId:t,options:e});const s=Ee.get(t).frames;return s.sessionId=t,s},e.initResponse=Re,Object.defineProperty(e,"__esModule",{value:!0})}));