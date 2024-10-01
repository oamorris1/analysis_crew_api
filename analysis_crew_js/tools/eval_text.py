

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings


 

from dotenv import load_dotenv, find_dotenv
from langchain.evaluation import load_evaluator


load_dotenv(find_dotenv('.env'))
embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", model="text-embedding-ada-002", chunk_size=10)
deployment_name3 = "gpt-35-turbo-16k"
deployment_name4 = "gpt-4"
llm_gpt3 = AzureChatOpenAI(deployment_name=deployment_name3, model_name=deployment_name3, temperature=0, streaming=True)
llm_gpt4 = AzureChatOpenAI(deployment_name=deployment_name4, model_name=deployment_name4, temperature=0, streaming=True)
accuracy_criteria = {
    "accuracy": """
Score 1: The answer is completely unrelated to the reference.
Score 3: The answer has minor relevance but does not align with the reference.
Score 5: The answer has moderate relevance but contains inaccuracies.
Score 7: The answer aligns with the reference but has minor errors or omissions.
Score 10: The answer is completely accurate and aligns perfectly with the reference."""
}

text1 = """
The Denver Zoological Foundation provides information on the African lion (Panthera leo melanochaita), a vulnerable species primarily found
 in sub-Saharan Africa's savannas, grasslands, and woodlands. Classified as a mammal within the order Carnivora and the family Felidae, lions 
 are the second-largest big cat species, with males larger than females and possessing distinctive manes. They are social animals, living in 
 prides and exhibiting communal hunting and parenting behaviors. Lions are carnivorous, preying on various ungulates and scavenging when necessary.
In captivity, their diet is carefully managed with nutrient-fortified meat and occasional fasting.Lions have several adaptations for survival,
including group behavior, night vision, and powerful bodies with retractable claws for hunting.
They are apex and keystone predators, with roars that can be heard up to 5 miles away. 
The Denver Zoo houses a coalition of four male lions born in 2015 and a pride with several females and a male cub born in 2019.
The lions' conservation status is vulnerable due to habitat loss and human conflict, with an estimated 43% population reduction over 21 years,
leaving around 20,000 lions in the wild. Conservation efforts focus on creating safe habitats and national parks to ensure their survival.
Additional information can be found through resources like National Geographic and the Lion Recovery Fund.

"""

text2 = """
This systematic review by Daniel Martinez-Marquez and colleagues examines the application of eye tracking technology in high-risk industries such as aviation, maritime, and construction. The review highlights that most accidents in these sectors are due to human error, often linked to impaired mental performance and attention failure. Eye tracking research, dating back 150 years, captures a variety of eye movements that reflect human cognitive, emotional, and physiological states, providing insights into the human mind in different scenarios.

The review identifies the demographic distribution and applications of eye tracking research, revealing that the aviation industry has the highest number of studies, followed by maritime and construction. The USA leads in eye tracking studies, with significant contributions from Germany, Norway, China, and the UK. The research uncovers different applications of eye tracking, such as studying visual attention, mental workload, human-machine interfaces, situation awareness, training improvement, and hazard identification.

Eye tracking technologies are often integrated with simulators, video and audio recording, head trackers, EEG, ECG, and other technologies to study various human aspects in detail. The review identifies gaps in the literature, suggesting the need for further research on topics like mental workload in construction, hazard detection in aviation and maritime, and the integration of additional technologies to support eye tracking research. The study concludes that eye tracking has a promising future in enhancing understanding and training in high-risk industries.
"""

text3 = """
The study, published in Transportation Engineering 13 (2023), examines the role of human factors in aviation ground operation-related accidents and incidents using a human error analysis approach. The research analyzed 87 accident and incident reports from 2000 to 2020, employing the Human Factors Dirty Dozen (HF DD) Model and the Human Factors Analysis and Classification Scheme (HFACS) for systematic thematic analysis. The findings highlight that the main causes of ground operation-related accidents and incidents are lack of situational awareness and failure to follow prescribed procedures. Critical operational actions identified include aircraft pushback/towing, aircraft arrival and departure, and aircraft weight and balance. The study proposes an agenda for future research and recommendations for industry corrective action, emphasizing the need for a comprehensive Ramp Resource Management (RRM) framework to address the identified safety issues. The research also suggests that current human error analysis models may need to be extended to consider the broader organization and aviation system context.

"""

text4 = " What are variables in accidents when people are flying and when they are driving cars"

text5 = "This paper by Omar A. Morris and Khalid H. Abed from Jackson State University describes the process of configuring and administering a Cray CS 400 heterogeneous cluster using Bright Cluster Manager software. The paper outlines the steps from hardware assembly to software installation, including the OS and system management tools. The Cray CS 400 cluster consists of a head node and six compute nodes, equipped with Intel Xeon CPUs and Intel Xeon Phi coprocessors, which are managed through a network topology that includes Ethernet and InfiniBand connections. The authors detail the installation of Bright Cluster Manager on a bare metal system and the challenges faced during the add-on installation method. They emphasize the importance of using a package installer and allowing Bright Cluster Manager to resolve any system conflicts. The paper also discusses the registration with Red Hat Network for OS distribution access, the creation of a custom compute node image, and the installation of the Many Integrated Core (MIC) software stack necessary for the Xeon Phi coprocessors. The research component of the paper focuses on evaluating the performance of the miniMD mini-application from the Mantevo Project on the cluster. The miniMD, a proxy for the LAMMPS program, is used to compare the performance of computations on a conventional multi-core Xeon processor with those on the Intel Xeon Phi architecture. The paper concludes by acknowledging the support received for the project and stating that future work will detail the cluster's performance and improvements to the installation and administration process."

text6 = """This thesis by Alexandra Levin and Najda Vidimlic at Mälardalen University, Sweden,
focuses on improving situational awareness in aviation through robust vision-based detection of hazardous objects.
The research evaluates the use of deep learning object detection algorithms, particularly Faster RCNN with ResNet-50-FPN,
to detect specific objects during the final approach to an airport. The study involves constructing
a comprehensive dataset that describes the operational environment, including various environmental conditions such as weather
and lighting changes. The dataset is extended through image collection, augmentation, and manual annotation. The object detector's
accuracy is assessed before and after introducing representations of environmental conditions into the training data.
Bayesian uncertainty estimations are also evaluated to determine if they can provide additional information for interpreting objects correctly
and detecting erroneous predictions. The results show that introducing variations of environmental conditions in the training set 
improves the robustness necessary to maintain accuracy when exposed to different environmental conditions. However, further research is
needed to conclude if uncertainty estimations can detect erroneous predictions effectively"""

text7 = "What are some common variables used in studies regarding human error-based accidents"

text8 = "What are some common variables used in studies regarding human error-based aviation accidents"


evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embeddings)
evaluator2 = load_evaluator("labeled_score_string", criteria=accuracy_criteria, llm=llm_gpt4)

eval_result_text1 = evaluator.evaluate_string_pairs(
    prediction="What are some common variables used in studies regarding human error-based aviation accidents", prediction_b=text1
)

eval_result_text2 = evaluator.evaluate_string_pairs(
    prediction="What are some common variables used in studies regarding human error-based aviation accidents", prediction_b=text2
)

eval_result_text3 = evaluator.evaluate_string_pairs(
    prediction="What are some common variables used in studies regarding human error-based aviation accidents", prediction_b=text3
)

eval_result_text4 = evaluator.evaluate_string_pairs(
    prediction="What are some common variables used in studies regarding human error-based aviation accidents", prediction_b=text4
)

eval_result_text5 = evaluator.evaluate_string_pairs(
    prediction="What are some common variables used in studies regarding human error-based aviation accidents", prediction_b=text5
)

eval_result_text6 = evaluator.evaluate_string_pairs(
    prediction="What are some common variables used in studies regarding human error-based aviation accidents", prediction_b=text6
)

eval_result_text7 = evaluator.evaluate_string_pairs(
    prediction="What are some common variables used in studies regarding human error-based aviation accidents", prediction_b=text7
)
eval_result_text8 = evaluator.evaluate_string_pairs(
    prediction="What are some common variables used in studies regarding human error-based aviation accidents", prediction_b=text8
)

eval2_result = evaluator2.evaluate_strings(
    prediction="What are some common variables used in studies regarding human error-based aviation accidents",
    reference=text2,
    input="Can the predidction be answered by the reference?",
)


print("result for text1 Lions: ",eval_result_text1)
print("result for text2 Application of Eye tracking: ",eval_result_text2)
print("result for text3 Role of Human Factors: ",eval_result_text3)
print("result for text4 Statement: ",eval_result_text4)
print("result for text5 HPC : ",eval_result_text5)
print("result for text6 Situ Awareness : ",eval_result_text6)
print("result for text7 statemet2 : ",eval_result_text7)
print("result for text8 exact : ",eval_result_text8)
print("eval2_result: ",eval2_result)



#single file uploads
# def handle_file_upload(event):
#     uploaded_file = file_input.value
#     file_name = file_input.filename 
#     save_folder_path = "C:/Users/Admin/Desktop/erdcDBFunc/analysis_crew/test"
#     save_path = Path(save_folder_path, file_name)
#     if uploaded_file:
#         with open(save_path, mode='wb') as w:
#             w.write(uploaded_file)
#         # Process the uploaded file
#         if save_path.exists():
#             chat_interface.send(f"File '{file_name}' uploaded successfully!", user="System", respond=False)


# def handle_file_upload(event):
#     uploaded_files = file_input.value
#     filenames = file_input.filename 
#     save_folder_path = "C:/Users/Admin/Desktop/erdcDBFunc/analysis_crew/test"
    
#     if uploaded_files:
#         for file_content, file_name in zip(uploaded_files, filenames):
#             save_path = Path(save_folder_path, file_name)
#             with open(save_path, mode='wb') as w:
#                 w.write(file_content)
#         # Here you can process the uploaded file
#             if save_path.exists():
#                 chat_interface.send(f"File '{file_name}' uploaded successfully!", user="System", respond=False)
