# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 22:54:37 2016

@author: johannes
"""

class country:
    def __init__(self):
        self.cities=[]
        
    def addCity(self,city):
        self.cities.append(city)
        

class city:
    def __init__(self,numPeople):
        self.people=[]
        self.numPeople=numPeople
        
        
    def addPerson(self,person):
        self.people.append(person)
    
    def join_country(self,country):
        self.country=country
        country.addCity(self)
        
        for i in range(self.numPeople):
                person(i).join_city(self)
      
class person:
    def __init__(self,ID):
        self.ID=ID

    def join_city(self,city):
        self.city=city
        city.addPerson(self)
        
    def people_in_my_country(self):
        x= sum([len(c.people) for c in self.city.country.cities])
        return x
        
US=country()
NYC=city(10).join_country(US)
SF=city(5).join_country(US)

print US.cities[0].people[0].people_in_my_country()

# 15