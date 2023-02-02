


if __name__ == '__main__':
    import argparse
    from subprocess import Popen, PIPE, check_call

    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=str, default='https://www.wbtv.com/')
    parser.add_argument('--from_date', type=str, default='20230101')
    parser.add_argument('--to_date', type=str, default='20230102')
    args = parser.parse_args()

    output, err = Popen([
        "waybackpack", args.site,
        "--from-date",
        args.from_date,
        "--to-date",
        args.to_date,
        "--list"
    ], stdin=PIPE, stdout=PIPE, stderr=PIPE).communicate()

    wayback_urls = output.decode().split()

    for url in wayback_urls:
        output = check_call(f'wget --no-clobber --page-requisites --convert-links --domains web.archive.org --no-parent {url}', shell=True)