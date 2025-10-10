## Smartphone Market Analysis using Web Scraping - Flipkart Case Study

### 

#### Project Overview



This project focuses on extracting **real-time smartphone data from Flipkart** using Python web scraping techniques (BeautifulSoup, Requests). The smartphone industry is highly competitive, with frequent product launches, fluctuating prices, and diverse customer preferences.



The dataset includes product names, prices, discounts, ratings, and technical specifications, which are further cleaned and prepared for analysis.



The ultimate goal is to visualize market trends in Power BI, providing insights into -

* Price distribution across smartphone models
* Discount patterns for different brands
* Consumer rating behavior
* Key feature comparisons (RAM, Battery, Display, Processor, Camera)



This project simulates an end-to-end data analysis workflow — from raw web scraping to structured data analysis — showcasing practical skills in data collection, cleaning, transformation, and visualization.



---



#### Data Source



\- **Website -** Flipkart (https://www.flipkart.com/)

\- **Target Data -** Smartphone listings

  - Brand and model

  - Price and original price

  - Discount percentage

  - Ratings

  - Key specifications (RAM, ROM, Display size, Battery, Camera, Processor)



---



#### Tools \& Libraries



\- **Python** (programming language)  

\- **Requests** (for sending HTTP requests)  

\- **BeautifulSoup** (for HTML parsing)  

\- **Pandas** (for data storage and initial handling)



---



#### Methodology



###### Step 1: Setting up the environment

Imported the required libraries and defined the base URL for Flipkart smartphone searches.



###### Step 2: Identifying the target website \& structure

Analyzed Flipkart’s smartphone search page structure using browser developer tools. Identified HTML tags and classes for:

\- Product name  

\- Price  

\- Ratings  

\- Specifications  



###### Step 3: Scraping process

\- Sent HTTP requests with headers (including User-Agent) to avoid blocking.  

\- Parsed the HTML content using BeautifulSoup.  

\- Extracted relevant fields into Python lists.



###### Step 4: Pagination handling

\- Flipkart search results span multiple pages.  

\- Implemented a loop with page numbers to collect data across multiple result pages.



###### Step 5: Storing scraped data

\- Structured the extracted information into a Pandas DataFrame.  

\- Exported the dataset to `flipkart\_smartphones.csv` for further cleaning and analysis.



---



#### Challenges \& Considerations



While scraping, the webpage displayed 9,284 products on the selected page. However, after executing the scraping script, only 984 product records were successfully extracted. This discrepancy highlights potential challenges such as:



* **Pagination Handling:** The scraper may not have been configured to navigate through all product pages.
* **Dynamic Loading (JavaScript/AJAX):** Some product data might load dynamically as the user scrolls, which standard scraping techniques may miss.
* **Anti-Scraping Mechanisms:** The website could have protections (e.g., rate limits, bot detection, hidden elements) preventing full data extraction.
* **Selector Limitations:** The chosen HTML tags or CSS selectors might not capture all product listings.
* **Ethical considerations:** Scraping should respect website terms of service.



These factors need to be carefully addressed to ensure the scraper retrieves the complete dataset rather than just a partial subset.



---



#### Closing Note

The web scraping phase successfully provided a raw dataset of smartphones from Flipkart. This dataset will now be refined and explored in the **Data Cleaning \& Transformation** stage to uncover meaningful market insights.









































