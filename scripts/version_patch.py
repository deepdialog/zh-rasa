#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re


def main():
    path = 'zh_rasa/version.txt'
    version = open(path, 'r').read().strip()
    print(version)
    assert re.match(r'^\d+\.\d+\.\d+$', version)
    a, b, c = [int(x) for x in version.split('.')]
    c += 1
    new_version = '{}.{}.{}'.format(a, b, c)
    print(new_version)
    with open(path, 'w') as fp:
        fp.write(new_version)


if __name__ == '__main__':
    main()
