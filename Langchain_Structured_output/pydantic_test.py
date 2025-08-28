from pydantic import BaseModel

# Example of using Pydantic to validate data
# This will raise a validation error because 'name' should be a string, not an integer
class student(BaseModel):
    name: str
 

# student_data = {'name':32}   # Uncommenting the above line will raise a validation error
student_data = {'name': 'John Doe'}  # Correct data type
student1=student(**student_data)
print(student1)


# now example of how we can use pydantic and typing->Optional to make a field optional
from typing import Optional
class student2(BaseModel):
    name: Optional[str] = None  # This field is optional
student_data2 = {'name': None}
student2_instance = student2(**student_data2)
print(student2_instance)

# Example of using Pydantic for Email validation
from pydantic import EmailStr
class User(BaseModel):
    email: EmailStr  # This will validate that the email is in the correct format
# user_data = {'email': 'abc'}  # This will raise a validation error because it's not a valid email
user_data = {'email': 'abc@example.com'}  # Correct email format
user_instance = User(**user_data)  # Uncommenting this line will raise a validation
print(user_instance)

from pydantic import Field
class User(BaseModel):
    email: EmailStr  # This will validate that the email is in the correct format
    cgpa:float = Field(ge=0.0, le=10.0, default=7)  # This will validate that the CGPA is between 0.0 and 10.0
  
# user_data = {'email': 'abc'}  # This will raise a validation error because it's not a valid email
user_data = {'email': 'abc@example.com'}  # Correct email format
user_instance = User(**user_data)  # Uncommenting this line will raise a validation
print(user_instance.cgpa)


# now examples of pydantic fields functions
