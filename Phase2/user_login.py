import json

class User:
    def __init__(self, user_filepath, driver):
        with open(user_filepath, 'r') as f:
            data=json.load(f)
            self.userid=data['linkedin_username']
            self.password=data['linkedin_password']
            self.driver=driver

    def sign_in(self, url):

        self.driver.get(url)
        self.driver.maximize_window()
        signin_page = self.driver.find_element("xpath", ".//a[contains(text(), 'Sign in') and @href]")
        signin_page.click()
        username_field = self.driver.find_element("xpath" , ".//input[@id = 'username']")
        password_field = self.driver.find_element("xpath" , ".//input[@id = 'password']")
        submit_button = self.driver.find_element("xpath" , ".//button[@type = 'submit']")
        
        username_field.send_keys(self.userid)
        password_field.send_keys(self.password)
        submit_button.click()


    def search_jobs(self):
        self.driver.get_url("https://www.linkedin.com/jobs")
        self.driver.maximize_window()

        role_input = self.driver.find_element("xpath" , ".//input[@class = 'jobs-search-box__text-input jobs-search-box__keyboard-text-input jobs-search-global-typeahead__input']")
        role_input.send_keys(role)
        driver.implicitly_wait(2)
        location_input = driver.find_element("xpath" , ".//input[@class = 'jobs-search-box__text-input' and @autocomplete=  'address-level2']")
        location_input.send_keys(location)
        #submit
        role_input.send_keys(Keys.ENTER)


    def applyfilters():
        return 0


    



