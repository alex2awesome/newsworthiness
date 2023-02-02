import typing
from pathlib import Path

import click
from playwright.sync_api import sync_playwright
from playwright.sync_api._generated import BrowserContext
from retry import retry
import os
here = os.path.dirname(__file__)

instantiate_model_js = '''
    var predictor = new LRUrlPredictor(%s)
'''

instantiate_heuristic_js = '''
    var predictor = new HueristicUrlPredictor()
'''

get_link_divs_js = '''
    var a_counts = {}
    var as = []
    var a_top_nodes = Array.from(document.querySelectorAll('a'))
            //filter out null links
            .filter(function(a) { return a.href !== ''})
            .filter(function(a){return a.href !== undefined; })
            // predict whether the URL is an article URL are not
            .filter(function(a){return predictor.get_prediction(a.href)})
            // process
            .map(function(a, i) {
                a_counts[a.href] = a_counts[a.href] || []
                a_counts[a.href].push(i)
                return a
            } )
            .map(function(a, i, as_arr){
                as.push(a)
                return get_highest_singular_parent(i, as_arr, a_counts) 
            })     
'''

js_to_spotcheck = '''
    a_top_nodes.forEach(function(node){
        node.setAttribute('style', 'border: 4px dotted blue !important;')
    })
'''


async def add_visual_bounding_boxes(page):
    await page.evaluate('''
        () => a_top_nodes.map( (a) => a.setAttribute('style', 'border: 4px dotted blue !important;') )
    ''')


# load helper scripts into the page and get resources to run the rest of the scripts
async def load_model_files_and_helper_scripts(page):
    """Read and return Javascript code from a file. Convenience function."""
    model_utils_script = os.path.join(here, "js", "model_utils.js")
    with open(model_utils_script) as f:
        await page.evaluate(f.read())

    utils_script = os.path.join(here, "js", "utils.js")
    with open(utils_script) as f:
        await page.evaluate(f.read())

    model_weights = os.path.join(here, 'js', 'trained_lr_obj.json')
    with open(model_weights) as f:
        return f.read()


async def get_bounding_box_info(page):
    bounding_boxes = await page.evaluate('''
        function () {
            var all_links = []
            a_top_nodes.forEach(function(node){
                var links = Array.from(node.querySelectorAll('a'))
                if ((links.length == 0) & (node.nodeName === 'A')){
                    links = [node]
                }
                
                var seen_links = {};
                links = links
                    .map(function(a) {return {
                        'href': a.href,
                         'link_text' : get_text_of_node(a), 
                         'is_article': predictor.get_prediction(a.href)
                        }
                    } )
                    .filter(function(a){return a.is_article})
                    .sort((a, b) => { return  b.link_text.length - a.link_text.length } )
                    .filter(function(a){
                        if (!(a.href in seen_links)) {
                            seen_links[a.href] = true;
                            return true
                        }
                        return false 
                    })
                    .forEach(function(a){
                        var b = node.getBoundingClientRect() // get the bounding box around the entire defined node.
                        a['x'] = b['x']
                        a['y'] = b['y']
                        a['width'] = b['width']
                        a['height'] = b['height']
                        a['all_text'] = get_text_of_node(node)
                        all_links.push(a)
                })
            })
            
            seen_all_links = {}
            return all_links.filter(function(a){
                if (!([a.href, a.x, a.y] in seen_all_links)) {
                    seen_all_links[[a.href, a.x, a.y]] = true;
                    return true;
                }
                return false;
            })
        }
    ''')

    width = await page.evaluate('''
        Math.max(
            document.documentElement["clientWidth"],
            document.body["scrollWidth"],
            document.documentElement["scrollWidth"],
            document.body["offsetWidth"],
            document.documentElement["offsetWidth"]
        );
    ''')

    height = await page.evaluate('''Math.max(
        document.documentElement["clientHeight"],
        document.body["scrollHeight"],
        document.documentElement["scrollHeight"],
        document.body["offsetHeight"],
        document.documentElement["offsetHeight"]
    );''')

    return {'bounding_boxes': bounding_boxes, 'width': width, 'height': height}