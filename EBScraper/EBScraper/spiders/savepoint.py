from __future__ import absolute_import
import scrapy
from ..items import EbscraperItem

# Follow the instructions given, BUT, set it up to run all of the URLs at once
# What is needed: The domain, the restricted domain (what is the longest URL that every post has in common), the element that has the text
# What can my collaborators do? Get these things, and then scrape, but more importantly get these things
# If they can get these things, we can make it a dictionary or list of lists and have it be very automoated right out of the gate


class EchoBurst_Spider(scrapy.Spider):
    # spider name
    name = "savePoint"
    # MUST NOT HAVE HTTP or WWW
    allowed_domains = ["realclimate.org/"]
    # start_urls - crawling single or multiple websites , list formate
    start_urls = ["http://www.realclimate.org/"]

    def parse(self, response):
        for sel in response.css('body'):
            items = EbscraperItem()
            items["contents"] = sel.css('p').extract()
            items["contents"] = " ".join(items["contents"]).split("</p>")
            yield items

        next_page = response.css('#content > div.navigation > div.alignleft > a::attr(href)').extract_first()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)
