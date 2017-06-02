from __future__ import absolute_import
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from lxml import html
# Use this to run from terminal (cd into the first EBScraper folder and enter scrapy crawl EBSpider - o test.jl 
from ..items import EbscraperItem
# from EBScraper.EBScraper.items import EbscraperItem

# Follow the instructions given, BUT, set it up to run all of the URLs at once
# What is needed: The domain, the restricted domain (what is the longest URL that every post has in common), the element that has the text
# What can my collaborators do? Get these things, and then scrape, but more importantly get these things
# If they can get these things, we can make it a dictionary or list of lists and have it be very automoated right out of the gate


class EchoBurst_Spider(CrawlSpider):
    # spider name
    name = "EBSpider"
    # MUST NOT HAVE HTTP or WWW
    allowed_domains = ["skepticalscience.com"]
    # start_urls - crawling single or multiple websites , list formate
    start_urls = ["https://www.skepticalscience.com/anti-vax-climate-gmo-all-same.html"]

    def parse(self, response):
        for sel in response.css('body'):
            items = EbscraperItem()
            items["contents"] = sel.css('p::text').extract()
            # Will have to add a comment break, to ensure comments aren't included
            items["contents"] = " ".join(items["contents"]).split("</p>")
            yield items
        # Currently non-functional
        # next_page = response.css('#content > div.navigation > div.alignleft > a::attr(href)').extract_first()
        # if next_page is not None:
        #     yield response.follow(next_page, callback=self.parse)
