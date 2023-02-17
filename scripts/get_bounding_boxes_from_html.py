import os
from playwright.async_api import async_playwright
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from datetime import datetime
from tqdm.auto import tqdm
import os
import pandas as pd
import re
import numpy as np
import math
from collections import defaultdict
from more_itertools import flatten


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
async def async_load_model_files_and_helper_scripts(page):
    """Read and return Javascript code from a file. Convenience function."""
    model_utils_script = os.path.join(here, "js", "model_utils.js")
    with open(model_utils_script) as f:
        await page.evaluate(f.read())

    utils_script = os.path.join(here, "js", "utils.js")
    with open(utils_script) as f:
        await page.evaluate(f.read())

    url_parsing_script = os.path.join(here, 'js', 'psl.min.js')
    with open(url_parsing_script) as f:
        await page.evaluate(f.read())

    model_weights = os.path.join(here, 'js', 'trained_lr_obj.json')
    with open(model_weights) as f:
        return f.read()


async def async_get_bounding_box_info(page):
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


# load helper scripts into the page and get resources to run the rest of the scripts
def load_model_files_and_helper_scripts(page):
    """Read and return Javascript code from a file. Convenience function."""
    model_utils_script = os.path.join(here, "js", "model_utils.js")
    with open(model_utils_script) as f:
        page.evaluate(f.read())

    utils_script = os.path.join(here, "js", "utils.js")
    with open(utils_script) as f:
        page.evaluate(f.read())

    url_parsing_script = os.path.join(here, 'js', 'psl.min.js')
    with open(url_parsing_script) as f:
        page.evaluate(f.read())

    model_weights = os.path.join(here, 'js', 'trained_lr_obj.json')
    with open(model_weights) as f:
        return f.read()


def get_bounding_box_info(page):
    bounding_boxes = page.evaluate('''
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

    width = page.evaluate('''
        Math.max(
            document.documentElement["clientWidth"],
            document.body["scrollWidth"],
            document.documentElement["scrollWidth"],
            document.body["offsetWidth"],
            document.documentElement["offsetWidth"]
        );
    ''')

    height = page.evaluate('''Math.max(
        document.documentElement["clientHeight"],
        document.body["scrollHeight"],
        document.documentElement["scrollHeight"],
        document.body["offsetHeight"],
        document.documentElement["offsetHeight"]
    );''')

    return {'bounding_boxes': bounding_boxes, 'width': width, 'height': height}


class SorterAndFixer:
    @staticmethod
    def centerXY(xylist, how='center'):
        x, y = zip(*xylist)
        l = len(x)
        if how == 'center':
            return sum(x) / l, sum(y) / l
        if how == 'left': # get center-point on the left side
            return min(x), sum(y) / l
        if how == 'right': # get center-point on the right side
            return max(x), sum(y) / l
        if how == 'top': # get center-point on the top side
            return sum(x) / l, max(y)
        if how == 'bottom': # get center-point on the top side
            return sum(x) / l, min(y)

    @staticmethod
    def sort_points(xylist, cx=None, cy=None, how='center'):
        if (cx is None) and (cy is None):
            cx, cy = SorterAndFixer.centerXY(xylist, how=how)
        xy_sorted = sorted(xylist, key=lambda x: math.atan2((x[1] - cy), (x[0] - cx)))
        return xy_sorted

    @staticmethod
    def sort_two_rects(rect1, rect2, orient_rel_to_1='right', fix=False):
        def get_width_height_from_border(border):
            xs = list(map(lambda x: x[0], border))
            ys = list(map(lambda x: x[1], border))
            return max(xs) - min(xs), max(ys) - min(ys)

        def get_size_from_border(border):
            width, height = get_width_height_from_border(border)
            return width * height

        if orient_rel_to_1 == 'right':
            border_1 = SorterAndFixer.sort_points(rect1, how='right')
            border_2 = SorterAndFixer.sort_points(rect2, how='left')
            return border_1 + border_2

    @staticmethod
    def fix(xylist):
        output = []
        for p1, p2 in list(zip(xylist[:-1], xylist[1:])):
            output.append(p1)
            if not ((p1[0] == p2[0]) or (p1[1] == p2[1])):
                if abs(p1[0] - p2[0]) < abs(p1[1] - p2[1]):
                    new_point = (p1[0], p2[1])
                else:
                    new_point = (p2[0], p1[1])
                output.append(new_point)
        output.append(p2)
        return output


# pairs that are on the same y-grid
def is_adjacent(bb_df, idx_pair, tolerance=100):
    return (
        bb_df
        .loc[idx_pair][['x', 'y']]
        .apply(lambda x: abs(x.iloc[0] - x.iloc[1]) < tolerance)
        .any()
    )


def build_adjacency_index_list(bb_df, tolerance=100, return_how=False):
    def build_1d_adjacency(bb_df, matches, direction='x', return_how=False):
        sort_dir = ['x', 'y'] if direction == 'x' else ['y', 'x']
        first_ord = 'top' if direction == 'x' else 'left'
        second_ord = 'bottom' if direction == 'x' else 'right'
        idx_list = list(bb_df.sort_values(sort_dir).index)
        idx_pairs = list(zip(idx_list[:-1], idx_list[1:]))
        idx_pairs = list(map(list, idx_pairs))
        # get all pairs of article boxes that are less than `tolerance` pixels apart
        idx_pairs = list(filter(lambda idx_pair:
                                bb_df.loc[idx_pair][direction]
                                .pipe(lambda s: abs(s.iloc[0] - s.iloc[1]) < tolerance),
                                idx_pairs))
        # for each on of these pairs, check to see if the URL is the same
        for idx_pair in idx_pairs:
            url_key = 'site_url' if 'site_url' in bb_df else 'href'

            if bb_df.loc[idx_pair[0], url_key] == bb_df.loc[idx_pair[1], url_key]:
                if not return_how:
                    matches[idx_pair[0]].add(idx_pair[1])
                    matches[idx_pair[1]].add(idx_pair[0])
                else:
                    matches[idx_pair[0]].add((idx_pair[1], first_ord))
                    matches[idx_pair[1]].add((idx_pair[0], second_ord))
        return matches

    matches = defaultdict(set)
    matches = build_1d_adjacency(bb_df, matches, direction='x', return_how=return_how)
    matches = build_1d_adjacency(bb_df, matches, direction='y', return_how=return_how)
    return {k: list(matches[k]) for k in sorted(matches)}


def get_borders_for_one_adjacency_list(
        bb_df, idx_list, x_key='x', y_key='y', w_key='width', h_key='height', fix_points=True
):
    def _helper_get_border(x):
        return [(x[x_key], x[y_key]), (x['end_x'], x[y_key]), (x['end_x'], x['end_y']), (x[x_key], x['end_y'])]

    all_x_y_pairs = bb_df.loc[list(idx_list)][[x_key, y_key, w_key, h_key]]
    xy_pairs = (
        all_x_y_pairs
            .assign(end_x=lambda df: df[x_key] + df[w_key])
            .assign(end_y=lambda df: df[y_key] + df[h_key])
            .assign(borders=lambda df: df.apply(_helper_get_border, axis=1))
    )
    return SorterAndFixer.sort_two_rects(xy_pairs['borders'].iloc[0], xy_pairs['borders'].iloc[1], fix=fix_points)


def plot_bounding_box_df(
        bb_df, use_percs=False, url_to_filter=None, format_title_func=None, figsize=None, round_val=None,
        ax = None, clip_right=False, fix_adjacencies=False, adjancency_tolerance=100, colors=None,
):
    """
    Plots an abstracted representation of the bounding boxes

    :param bb_df: DataFrame representing a single webpage, where each row is an HTML div.
    :param url_to_filter: If provided, show the position of a single article on the webpage.
    :param format_title_func: function to format the title of the plot.
                                Useful, e.g. if the key is a datestring, then it can format it a bit nicer.
    :param figsize: figsize for the plot
    :return:
    """
    if format_title_func is None:
        format_title_func = lambda x: datetime.strptime(x, '%Y%m%d%H%M%S')

    if use_percs and ('norm_x' not in bb_df.columns):
        bb_df = normalize_x_y_height_width(bb_df, round_val=round_val)

    x_key = 'x' if not use_percs else 'norm_x'
    y_key = 'y' if not use_percs else 'norm_y'
    w_key = 'width' if not use_percs else 'norm_width'
    h_key = 'height' if not use_percs else 'norm_height'

    xmax = bb_df[x_key].max()
    ymax = bb_df[y_key].max()

    if clip_right:
        xmax = get_clip_size_grid(bb_df, perc_intersections_cutoff=.05)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.set_ylim((0, ymax))
    ax.set_xlim((0, xmax))

    adjacency_list = build_adjacency_index_list(bb_df, tolerance=adjancency_tolerance)
    seen_idx = set()

    for i, b in bb_df.iterrows():
        alpha = .2
        edgecolor = None

        # Color so that we can spot new and deleted articles.
        action = b.get('action')
        if action == 'added':
            facecolor = 'lightgreen'
        elif action == 'deleted':
            facecolor = 'pink'
        else:
            facecolor = 'lightblue'

        if colors is not None:
            facecolor = colors[i]

        # Focus on a specific URL.
        if (b.get('site_urls', None) == url_to_filter):
            alpha = .8
            edgecolor = 'black'

        patch = Rectangle(
            xy=(b[x_key], ymax - b[y_key] - b[h_key]),
            width=b[w_key],
            height=b[h_key],
            alpha=alpha,
            edgecolor=edgecolor,
            facecolor=facecolor
        )

        if fix_adjacencies:
            if i in seen_idx:
                continue

            if i in adjacency_list:
                idx_list = [i] + adjacency_list[i]
                borders = get_borders_for_one_adjacency_list(
                    bb_df, idx_list, x_key=x_key, y_key=y_key, w_key=w_key, h_key=h_key
                )
                patch = Polygon(
                    xy=borders,
                    alpha=alpha,
                    edgecolor=edgecolor,
                    facecolor=facecolor
                )
                seen_idx &= set(idx_list)

        ax.add_patch(patch)

    if not use_percs:
        page_width = bb_df['page_width'].drop_duplicates().iloc[0]
        page_height = bb_df['page_height'].drop_duplicates().iloc[0]
        ax.vlines(page_width, 0, ymax)
        ax.hlines(ymax - page_height, 0, xmax)

    ax.set_title(format_title_func(str(bb_df['key'].iloc[0])))
    return xmax, ymax, page_width, page_height


def normalize_x_y_height_width(bb_df, round_val=2, page_height=None, page_width=None):
    def round_or_none(s, round):
        if round is None:
            return s
        return s.round(round)

    def norm_dim(df, num_col, denom_col):
        return (df[num_col] / df[denom_col]).apply(lambda x: min(x, 1))

    return (
        bb_df
         .assign(norm_x=lambda df: df.pipe(norm_dim, 'x', 'page_width').pipe(round_or_none, round_val))
         .assign(page_height=lambda df: page_height or (df['y'] + df['height']).max())
         .assign(norm_y=lambda df: df.pipe(norm_dim, 'y', 'page_height').pipe(round_or_none, round_val))
         .assign(norm_width=lambda df: df.pipe(norm_dim, 'width', 'page_width').pipe(round_or_none, round_val))
         .assign(norm_height=lambda df: df.pipe(norm_dim, 'height', 'page_height').pipe(round_or_none, round_val))
    )


async def instantiate_new_page_object(headless=True):
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=headless)
    page = await browser.new_page()
    await page.route(
        "**/*",
        lambda route: route.abort()
        if route.request.resource_type == "image"
        else route.continue_()
    )

    await page.route(
        "https://web.archive.org*/*",
        lambda route: route.abort()
    )
    return page, browser, playwright



async def get_bounding_boxes_for_files(
        file_list, here='', headless=True, key_func=None, article_height_bins=None
):
    """
    Get bounding box information for a list of files on disk.

    :param file_list: The list of files you want to process.
    :param here: Provide this if you're not providing the full filepath in each item in `file_list`
    :param headless: Whether to run the browser headless or not (for debugging).
    :param key_func: Function to pull out the key for each file that you want to associate with the bounding box.
                     For backwards compatibilty, this is the greps out waybackmachine datestrings as keys.
    :return:
    """
    if key_func is None:
        key_func = lambda x: re.search('\d{14}', x)[0]

    #
    page, browser, playwright = await instantiate_new_page_object(headless=headless)

    #
    all_height_width = []
    all_bounding_box_dfs = []
    for one_file in tqdm(file_list):
        fp = os.path.join(here, one_file)
        file_key = key_func(fp)
        file = f'file://{fp}'

        await page.goto(file, timeout=0)
        # instantiate the model and the weights
        model_weights = await async_load_model_files_and_helper_scripts(page)
        await page.evaluate(instantiate_model_js % model_weights)
        await page.evaluate(get_link_divs_js)

        # bin article heights so we can make more meaningful comparisons.
        if article_height_bins is not None:
            await page.evaluate('''() => 
                a_top_nodes.map(function(d){
                    var binned_height = bin_number_to_array(d.offsetHeight, %s)
                    d.style.height = binned_height + 'px'
                    d.style.border = '0px',
                    d.style.padding = '0px'
                    d.style.margin = '0px'                                        
            })''' % article_height_bins)

        b = await async_get_bounding_box_info(page)

        all_height_width.append({
            'height': b['height'],
            'width': b['width'],
            'key': file_key
        })
        bounding_box_df = pd.DataFrame(b['bounding_boxes'])
        bounding_box_df['page_width'] = b['width']
        bounding_box_df['page_height'] = b['height']
        bounding_box_df['key'] = file_key
        all_bounding_box_dfs.append(bounding_box_df)

    await page.close()
    await browser.close()
    await playwright.stop()

    return all_bounding_box_dfs, all_height_width


def get_bin_counts_one_col(bb_df, colname, bins=None, bin_step_size=None):
    if bins is None:
        if 'norm' not in colname:
            end = int(bb_df[colname].max())
            if bin_step_size < 1:
                bin_step_size = int(end * bin_step_size)
        else:
            end = 1
        bins = np.arange(0, end + bin_step_size, step=bin_step_size)

    cut_counts = bb_df[colname].pipe(lambda s: pd.cut(s, bins=bins, right=False))
    return (
        cut_counts
        .apply(lambda x: x.left)
        .astype(float)
        .value_counts()
        .reindex(bins)
        .fillna(0)
        .astype(int)
    )


def transform_input_for_clustering(
        bb_df, bin_step_size=.05, use_x=True, use_y=True, use_width=True, use_height=True,
        recalculate_page_width=True, x_clip_perc=.05
):
    """Simple 1-d representation for x, y, widths, heights.

    * bb_df : the DF we wish to reason about.
    * bin_step_size: how to coarsify.
    * use_x: include coarsified `x` bincounts in the output vector.
    * use_y: include coarsified `y` bincounts in the output vector.
    * use_width: include coarsified `width` bincounts in the output vector.
    * use_height: include coarsified `height` bincounts  in the output vector.
    * recalculate_page_width: whether to recalculate the page width using a threshold for the number of divs.
    """

    bins = np.arange(0, 1 + bin_step_size, step=bin_step_size)
    if recalculate_page_width:
        bb_df['page_width'] = get_clip_size_grid(bb_df, perc_intersections_cutoff=x_clip_perc)

    if 'norm_x' not in bb_df.columns:
        bb_df = normalize_x_y_height_width(bb_df, round_val=None)

    all_counts = []
    num_digits = str(bin_step_size)[::-1].find('.')
    for colname, use_col in [
        ('norm_x', use_x), ('norm_y', use_y), ('norm_height', use_height), ('norm_width', use_width)
    ]:
        if use_col:
            binned_counts = get_bin_counts_one_col(bb_df, colname, bins)
            binned_counts = binned_counts.rename(lambda x: ('%s_%.' + str(num_digits) + 'f') % (colname, x))
            all_counts.append(binned_counts)

    return pd.concat(all_counts)


def get_coarsified_layout_grid(
    bb_df, max_width=None, max_height=None, y_step=.005, x_step=.005,
    show_progress=False, use_perc=None, clip_x=False, fill_with_ones=True,
    max_count = None
):
    """
    Coarsify the layout of the article divs.
    Returns a grid dataframe based on pixel space.

    * bb_df: the bounding box DF we wish to reason about.
    * max_width: If not none, this is the width of the page.
    * max_height: If not none, this is the height of the page.
    * y_step: how much to coarsify in the `y` direction.
    * x_step: how much to coarsify in the `x` direction.
    """
    bb_df = bb_df.assign(
        site_url=lambda df: df['href'].apply(lambda x: re.sub('https://web.archive.org/web/\d{14}/', '', x))
    )

    if use_perc:
        cols_to_iterate = ['norm_x', 'norm_y', 'norm_width', 'norm_height', 'site_url']
        if not bb_df.columns.isin(cols_to_iterate).all():
            bb_df = normalize_x_y_height_width(bb_df, round_val=None, page_width=max_width, page_height=max_height)
        max_width = 1
        max_height = 1

    else:
        max_width = max_width or (bb_df['x'] + bb_df['width']).max()
        if clip_x:
            max_width = get_clip_size_grid(bb_df)

        max_height = max_height or (bb_df['y'] + bb_df['height']).max()
        y_step = int(y_step * max_height)
        x_step = int(x_step * max_width)
        cols_to_iterate = ['x', 'y', 'width', 'height', 'site_url']


    y_bins = np.arange(0, max_height + y_step, y_step)
    x_bins = np.arange(0, max_width + x_step, x_step)

    grid = pd.DataFrame(index=y_bins, columns=x_bins)
    if fill_with_ones:
        grid = grid.fillna(0)

    if show_progress:
        iterable = tqdm(bb_df[cols_to_iterate].iterrows(), total=len(bb_df))
    else:
        iterable = bb_df[cols_to_iterate].iterrows()

    for _, (x, y, w, h, url) in iterable:
        if fill_with_ones:
            grid.loc[y: y + h, x: x + w] += 1
        else:
            grid.loc[y: y + h, x: x + w] = url
    if max_count is not None:
        grid = ( grid >= max_count ).astype(int)
    return grid


def get_clip_size_grid(
        bb_df=None, coarse_grid=None, pos_col='x', perc_step_size=.01, perc_intersections_cutoff=.05
):
    """
    Calculate `page_width` or `page_height` based on the number of article divs present
    a coarse grid layout (this stops us from having trailing divs throw off our sizing.)

    * bb_df: the bounding box DF we wish to reason about
    * coarse_grid: the precalculated coarsified grid
    * pos_col: which column to clip (either `x` or `y`)
    * perc_step_size:
            how granular we step from outwards in. A smaller `perc_step_size` means
            that we are calculating a more fine-grained stopping point.
    * perc_intersections_cutoff:
            the percentage of articles on the page that need to intersect with the stopping
            point for us to consider stopping.

    """
    if coarse_grid is None:
        coarse_grid = get_coarsified_layout_grid(bb_df, x_step=perc_step_size, y_step=perc_step_size)

    coarse_grid = coarse_grid if pos_col == 'x' else coarse_grid.T
    rev_cumsum = (
        coarse_grid
        .iloc[:, ::-1]
        .cumsum(axis=1)
        .pipe(lambda df: (df > 0).astype(int))
    )
    for col_idx in range(len(rev_cumsum.columns)):
        cutoff = rev_cumsum.columns[col_idx]
        if rev_cumsum[cutoff].mean() > perc_intersections_cutoff:
            return rev_cumsum.columns[col_idx - 1]


def restrict_to_wayback_urls_and_process(df):
    return (
        df
        .loc[lambda df: df['href'].notnull()]
        .loc[lambda df: ~df['href'].str.startswith('file:///')]
        .assign(site_url=lambda df: df['href'].apply(lambda x: re.sub('https://web.archive.org/web/\d{14}/', '', x)))
    )



def draw_line_across_axes(x_1, y_1, x_2, y_2, ax1, ax2):
    ax1.annotate(
        '',
        xy=(x_1, y_1),
        xytext=(x_2, y_2),
        xycoords=ax1.transData,
        textcoords=ax2.transData,
        arrowprops=dict(facecolor='black', arrowstyle='-', clip_on=False)
    )

    ax2.annotate(
        '',
        xy=(x_1, y_1),
        xytext=(x_2, y_2),
        xycoords=ax1.transData,
        textcoords=ax2.transData,
        arrowprops=dict(facecolor='black', arrowstyle='<-')
    )

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

def connect_rectangles_using_hungarian_matching(df):
    t = (df
     .assign(midpoints=lambda df: df.apply(lambda x:
         (
             (x['x_x'] + x['width_x'] / 2),  # x1
             (x['y_x'] + x['height_x'] / 2), # y1
             (x['x_y'] + x['width_y'] / 2),  # x2
             (x['y_y'] + x['height_y'] / 2)  # y2
         ), axis=1))
    )

    c = t.pivot(
        index='index_x',
        columns='index_y',
        values='midpoints',
    ).applymap(lambda x: cosine([x[0], x[1]], [x[2], x[3]]))

    cost_assignment = list(zip(*linear_sum_assignment(c)))
    idx_assignment = list(map(lambda x: (c.index[x[0]], c.columns[x[1]]), cost_assignment))

    return t.set_index(['index_x', 'index_y']).loc[idx_assignment].reset_index(drop=True)


def plot_merged_df(
        merged_bb_df, bb_1=None, bb_2=None, plot_individual=True, figsize=(12, 24), axarr=None,
        plot_arrows=True
):
    if plot_individual:
        if axarr is None:
            _, axarr = plt.subplots(1, 2, figsize=figsize)
        x1max, y1max, p1_w, p2_h = plot_bounding_box_df(bb_1, clip_right=True, ax=axarr[0], fix_adjacencies=False)
        x2max, y2max, p2_w, p2_h = plot_bounding_box_df(bb_2, clip_right=True, ax=axarr[1], fix_adjacencies=False)
        axarr[0].set_yticks([])
        axarr[1].set_yticks([])

    if plot_arrows:
        cols_of_interest = ['x_x', 'x_y', 'y_x', 'y_y', 'width_x', 'width_y', 'height_x', 'height_y']
        for i, b in merged_bb_df.iterrows():
            if ((b['x_x'] + b['width_x'] / 2) < p1_w) and ((b['x_y'] + b['width_y'] / 2) < p2_w):
                draw_line_across_axes(
                    b['x_x'] + b['width_x'] / 2,
                    y1max - b['y_x'] - b['height_x'] / 2,
                    b['x_y'] + b['width_y'] / 2,
                    y2max - b['y_y'] - b['height_y'] / 2,
                    axarr[0],
                    axarr[1]
                )


def merge_and_dedupe_bbs(bb_1, bb_2):
    bb_1 = restrict_to_wayback_urls_and_process(bb_1)
    bb_2 = restrict_to_wayback_urls_and_process(bb_2)

    bb_1 = (bb_1
            .assign(action=lambda df:
                    df['site_url']
                    .isin(bb_2['site_url'])
                    .map({True: 'same', False: 'deleted'}))
           )

    bb_2 = (bb_2
            .assign(action=lambda df:
                    df['site_url']
                    .isin(bb_1['site_url'])
                    .map({True: 'same', False: 'added'})
                   )
           )

    merged_bb_df = (
        bb_1
        .reset_index()
        .merge(
            bb_2.reset_index(),
            how='inner',
            right_on='site_url',
            left_on='site_url'
          )
    )
    merged_bb_df = (
        merged_bb_df
           .groupby('site_url')
           .apply(connect_rectangles_using_hungarian_matching)
    )
    return merged_bb_df, bb_1, bb_2


from matplotlib.patches import Rectangle
import numpy as np


def get_zero_ranges(a, threshold=.01):
    """Get the sequences in `a` that contain consecutive zeros.

    threshold
    """
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    threshold = int(len(a) * threshold)
    ranges = list(filter(lambda x: abs(x[1] - x[0]) > threshold, ranges))
    return ranges


def convert_to_grid_idx(x, g_idx, buffer=0):
    """get a buffered index."""
    full_len = len(g_idx)
    s_idx, e_idx = x
    s_idx = 0 if s_idx == 0 else s_idx + buffer
    e_idx = e_idx - buffer if e_idx != full_len else e_idx - 1
    return [g_idx[s_idx], g_idx[e_idx]]


def get_bands_for_each_article(bb_df):
    grid = get_coarsified_layout_grid(bb_df, clip_x=True)
    m = grid.mean(axis=1)
    g_idx = grid.index
    r_idxs = get_zero_ranges(m)
    segments = list(map(lambda x: convert_to_grid_idx(x, g_idx), r_idxs))
    bands = list(map(lambda x: sum(x) / 2, segments))
    return np.digitize(bb_df['y'], bands)


def get_min_max_height_for_each_band(bb_df):
    return (bb_df
            .assign(y_end=lambda df: df['y'] + df['height'])
            .groupby('bands')
            .agg({'y': 'min', 'y_end': 'max'})
            .rename(columns={'y': 'min', 'y_end': 'max'})
            )


def get_mean_start_stops_shifted(i, band_start_stops):
    mean_start_stops = (pd.concat(
        filter(lambda df: df.shape[0] == i, band_start_stops))
                        .groupby(level=0)
                        .mean()
                        )

    output = []
    prev_max = 0
    for i, (_min, _max) in mean_start_stops.iterrows():
        shift = _min - prev_max
        _min = prev_max
        _max = _max - shift
        prev_max = _max
        output.append({'min': _min, 'max': _max})
    return pd.DataFrame(output, index=mean_start_stops.index)


def resize_bb_df_by_bands(bb_df, mean_start_stops):
    one_df_bounds = get_min_max_height_for_each_band(bb_df)
    num_bands = one_df_bounds.index.shape[0]
    resize_df = (
        mean_start_stops[num_bands]
        .rename(columns={'min': 'mean_min', 'max': 'mean_max'})
        .merge(one_df_bounds, left_index=True, right_index=True)
        .assign(shift=lambda df: df['mean_min'] - df['min'])
        .assign(squeeze=lambda df: (df['mean_max'] - df['mean_min']) / (df['max'] - df['min']))
    )

    return (
        bb_df
        .copy()
        .merge(resize_df, left_on='bands', right_index=True)
        .assign(y=lambda df:
        (df['y'] - df['min']) * df['squeeze'] + df['min'] + df['shift'])
        .assign(height=lambda df: df['height'] * df['squeeze'])
    )

#             if i in adjacency_list:
#                 idx_list = [i] + adjacency_list[i]
#                 borders = get_borders_for_one_adjacency_list(
#                     bb_df, idx_list, x_key=x_key, y_key=y_key, w_key=w_key, h_key=h_key
#                 )
#                 patch = Polygon(
#                     xy=borders,
#                     alpha=alpha,
#                     edgecolor=edgecolor,
#                     facecolor=facecolor
#                 )
#                 seen_idx &= set(idx_list)