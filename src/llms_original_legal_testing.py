import argparse
import os
import sys

import tiktoken
import torch
import datasets
import time

import transformers

from tools import methods
from openai import OpenAI

TOKEN_LLAMA = ""
TOKEN_GPT = ""

parser = argparse.ArgumentParser(description='Process values.')
parser.add_argument('model_name', type=str, help="Name of the model")
parser.add_argument('dataset_name', type=str, help="Name of the dataset to use")
parser.add_argument('set', type=str, help="Set to train on between train, validation and test")
parser.add_argument('--dict', type=str, default='', help="Word file name for insertion")
parser.add_argument('--inter_dict', type=str, default="",
                    help="Second word file name for intersectional method")
parser.add_argument('--output_path', type=str, default="../output/", help="Path to the output folder")
parser.add_argument('--data_path', type=str, default="../data/", help="Path to the data folder")
parser.add_argument('--use_bfloat16', action='store_true', help="Enable usage of bfloat16 for the model instead of float16. Enable if your card support it.")
parser.add_argument('--use_float32', action='store_true', help="Enable usage of torch.float32. Use this only if you dont have float16 support and no gpus.")
parser.add_argument('--cpu', action='store_true', help="Enable usage of cpu only.")
parser.add_argument('--gpu', action='store_true', help="Put all processes on GPU.")

args = parser.parse_args()
model_name = args.model_name
dataset_name = args.dataset_name
word_file1 = args.dict
word_file2 = args.inter_dict
split_of_dataset = args.set
output_folder_path = args.output_path
data_path = args.data_path
bfloat = args.use_bfloat16
float32 = args.use_float32
cpu_only = args.cpu
gpu_only = args.gpu

datasets_names = ['ecthr_a', 'eurlex', 'scotus', 'ledgar', 'imdb', 'winogender']
models = ['llama2', 'falcon', "gpt"]
models_paths = ["meta-llama/Llama-2-7b-chat-hf", "tiiuae/falcon-7b-instruct", "gpt-3.5-turbo-0125"]
if dataset_name not in datasets_names:
    print("Unknown dataset")
    exit(1)

if model_name not in models:
    print("Unknown model")
    exit(1)
index_model = models.index(model_name)
index_dataset = datasets_names.index(dataset_name)

is_intersectional = True if word_file1 != "" and word_file2 != "" else False
is_atomic = True if word_file1 != "" and not is_intersectional else False
is_original_testing = True if not is_intersectional and not is_atomic else False

word_file_name = methods.getFileName(word_file1)
inter_file_name = methods.getFileName(word_file2)
is_insertion = True if dataset_name == "winogender" and "_" not in word_file_name else False

full_list_name = "original" if is_original_testing else word_file_name + (('+'+inter_file_name) if is_intersectional else "")


if dataset_name in ["imdb", "winogender"] : split_of_dataset = "None"
output_dir = output_folder_path + "/".join(["llm_testing", model_name, dataset_name,""]) + split_of_dataset +"_"+full_list_name+".pkl"

if os.path.isfile(output_dir):
    print("Files exist - Done and exiting.")
    exit(0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if cpu_only:device=torch.device('cpu')
################# Dataset #################################
if dataset_name == 'imdb':
    dataset_path = data_path+"/imdb/imdb.csv"
    content = methods.getFromCSV(dataset_path, delimiter=",", header=0)
    dataset_labels = [1 if x == 'positive' else 0 for x in content[:, 1]]
    dataset_text = content[:,0]
elif dataset_name == 'winogender':
    dataset_path = data_path + "/winogender/sentences.tsv"
    content = methods.getFromCSV(dataset_path, delimiter="\t")
    dataset_labels = [x.split('.')[2] for x in content[1:, 0]]
    nouns = [[x.split('.')[0], x.split('.')[1]]  for x in content[1:, 0]]
    dataset_text = content[1:, 1]
else :
    complete_dataset = datasets.load_dataset("lex_glue", dataset_name, split=split_of_dataset)
    complete_dataset_size = len(complete_dataset)
    dataset_labels_column_name = 'labels' if dataset_name in ["ecthr_a", "ecthr_b", "eurlex"] else 'label'
    dataset_labels = [complete_dataset[i][dataset_labels_column_name] for i in range(complete_dataset_size)]
    dataset_text = [''.join(complete_dataset[i][('text')]) for i in
                    range(complete_dataset_size)]
dataset_text = methods.format_dataset(dataset_text)
dataset_size = len(dataset_labels)

print(f"Using dataset named {dataset_name} {split_of_dataset} set of length {dataset_size}.\n"
      f"Using model {model_name} and doing method \'{'Original' if is_original_testing else 'Replacement'}\'.\n"
      f"Word file used is {full_list_name}.")

def getChatEcthr(case, label_names, give_task=True):
    # case1 = "4.  The applicant was born in 1968 and lives in İstanbul.5.  On 11 December 2000 the applicant was arrested on suspicion of fraud.6.  On 12 February 2001 he was released pending trial.7.  On 29 November 2005 the İstanbul Assize Court sentenced the applicant to two years and one month's imprisonment.8.  On 19 October 2006 the Court of Cassation upheld the judgment of 29 November 2005."
    # label1 = [3] #5891
    # case2 = "4.  The applicants’ dates of birth and places of residence are given in the Appendix.5.  The applicants were opposition-oriented activists. At the material time the first applicant was a member of one of the opposition parties, the Popular Front Party of Azerbaijan. He participated in a number of peaceful demonstrations organised by the opposition and on several occasions was arrested and convicted for that.6.  On 12 October 2013 an opposition group İctimai Palata held a demonstration, authorised by the relevant authority, the Baku City Executive Authority (“the BCEA”). The demonstration was held in the Mahsul stadium, a place proposed by the BCEA. It was intended to be peaceful and was conducted in a peaceful manner. The participants were protesting against alleged irregularities and fraud during the presidential elections of 9 October 2013.7.  All three applicants participated in that demonstration.8.  After the end of the demonstration of 12 October 2013 the applicants were arrested at the entrance to a nearby metro station, Inshaatchilar.9.  The circumstances of the applicants’ arrests, their custody and subsequent administrative proceedings against them are similar to those in Huseynli and Others v. Azerbaijan (nos. 67360/11 and 2 others, 11 February 2016) and Huseynov and Others v. Azerbaijan ([Committee] nos. 34262/14 and 5 others, 24 November 2016) (see also Appendix)."
    # label2 = [7, 2, 3] #9000 + 898
    # case3 = "5.  The applicant was born in 1955 and lives in Kaposvár.6.  The applicant suffers from a so-called psycho-social disability. On 3 October 2006 the Court found a violation of Article 5 § 1 concerning his psychiatric detention (see Gajcsi v. Hungary, no. 34503/03).7.  On 12 January 2000 the Fonyód District Court placed the applicant under partial guardianship. As an automatic consequence flowing from Article 70(5) of the Constitution, as in force at the relevant time, his name was deleted from the electoral register.8.  On 8 July 2008 the Kaposvár District Court reviewed the applicant’s situation and restored his legal capacity in all areas but health care matters. This change, however, did not restore his electoral rights.9.  The applicant could not vote in the general elections held in Hungary on 11 April 2010, due to the restriction on his legal capacity which resulted in the deprivation of his electoral rights."
    # label3 = [] # 8217
    case1 = "5.  The applicant was born in 1960 and lives in Gaziantep.6.  On 14 January 1995 the applicant was taken into police custody by police officers from the Anti-Terror branch of the Izmir Security Directorate on suspicion of membership of an illegal organisation, namely the PKK (the Kurdistan Workers' Party).7.  On 23 January 1995 he was brought before a single judge of the Izmir State Security Court who ordered his detention on remand.8.  On 13 February 1995 the principal public prosecutor at the Izmir State Security Court filed a bill of indictment with the latter charging the applicant under Article 125 of the Criminal Code with carrying out activities for the purpose of bringing about the secession of part of the national territory.9.  On 2 December 1998 the Izmir State Security Court convicted the applicant as charged and sentenced him to death under Article 125 of the Criminal Code. Taking into account the applicant's behaviour during the trial, the death penalty was commuted to a life sentence.10.  On 17 April 2000 the Court of Cassation upheld the judgment of the Izmir State Security Court.11.  On 26 April 2000 the Court of Cassation's decision was pronounced in the absence of the applicant's representative.12.  On 1 June 2000 the Court of Cassation's decision was deposited with the registry of the First Instance Court.13.  On 21 December 2000 Law no. 4616, which governed the conditional release, suspension of proceedings or execution of sentences in respect of offences committed before 23 April 1999, came into force. The law stipulated that parole would not be applicable to persons who had committed offences under Article 125 of the Criminal Code. Thus, the applicant could not benefit from Law no. 4616."
    label1 = [3] #3178
    case2 = "7.  The applicant was born in 1957 and lives in Nicosia.8.  The applicant stated that he had been living in a house in Kyrenia and was the owner of a field with trees in the village of Ayios Epiktitos. As a result of the 1974 Turkish military intervention he had been deprived of his property rights, his property being located in the area which was under the occupation and the overall control of the Turkish military authorities. The latter had prevented him from having access to and using his house and property. He was being continuously prevented from entering the northern part of Cyprus because of his Greek-Cypriot origin.9.  The applicant produced two certificates from the Department of Lands and Surveys of the Republic of Cyprus, stating that he was the legal and registered owner of the following immovable property in the District of Kyrenia: -  Kyrenia, Ayios Epiktitos, Mevlitoudi, field with trees, sheet/plan 13/25, plot no. 120, surface: hectares 2, decares 2, sq. m 663, share: ¼.10.  The applicant acquired ownership of this property on 27 June 1990, when it was transferred to him by way of inheritance from his grandfather and a gift from his parents.11.  On 9 December 1990 the applicant made one further attempt to return to his property in northern Cyprus by participating in a convoy of cars of fellow refugees intending to return home during a peaceful march towards their villages. The applicant and his fellow refugees, who had informed the Commander of the United Nations (UN) forces in Cyprus of their intentions, stopped at the check point in the “buffer zone”, on the main road linking Nicosia with Ayios Amvrosios and Kyrenia. There, they asked the UN officer on duty to be allowed to return to their homes, property and villages. They requested the same officer to forward their demand to the Turkish military authorities. The officer replied that the latter had refused their request."
    label2 = [4, 9] #4184
    case3 = "4.  The applicant was born in 1941 and lives in Sosnowiec.5.  By a judgment of 8 August 2006 the Kraków Regional Administrative Court dismissed the applicant’s appeal against a second-instance administrative decision of 25 August 2005 by which the second-instance authority had refused to acknowledge that the applicant’s ailment was of an occupational character.6.  This judgment was served on the applicant on 25 September 2006.7.  By a decision of 19 October 2006 the court granted legal aid to the applicant.8.  The local Bar Association assigned a lawyer to the case by a decision of 2 November 2006. This decision was served on the lawyer on 8 November 2006. On 13 November 2006 the applicant gave a power of attorney to the lawyer.9.  On 1 December 2006 the lawyer submitted a cassation appeal to the Regional Administrative Court.10.  On 29 December 2006 that court held that the time-limit for lodging a cassation appeal had expired on 25 October 2006, thirty days after the applicant had been served with the judgment. The court rejected the cassation appeal on the ground that under the applicable laws the fact that the party was not represented by a lawyer and requested for legal aid only after having been served with the judgment had no bearing on the running of the time-limit for submitting a cassation appeal. It also noted that the legal-aid lawyer had failed to submit a request for leave to appeal out of time together with the cassation appeal.11.  The applicant’s lawyer appealed. He acknowledged that he had not submitted the request for leave to appeal out of time. However, he was of the view that this omission would not be relevant for the court’s decision to accept the cassation appeal for examination.12.  On 13 April 2007 the Supreme Administrative Court upheld the contested decision and shared the legal view expressed by the first-instance court."
    label3 = [] #6788

    #task = f"Provide the violated articles of the European Court of Human Rights articles ({', '.join(label_names)}), if any, given the factual paragraph." \
    #       " Answer as briefly as possible without explaining or phrasing. Only list the violated articles as follows: Article X, Article X, Article X, ..."
    #task =  #"Only and only respond with the following format if any case is violated: Article X, Article X, Article X, ... "\
    task = "Identify any violations of the specified European Court of Human Rights (ECTHR) articles ({', '.join(label_names)}) based on the provided factual paragraph."\
    " Only answer with the following format: Answer: Article X, Article X, Article X ..."
#" Only and only respond with following format: Article X, Article X, Article X, ..."
    question = "Which articles of the European Court of Human Rights are violated in this case, if any?"

    pairs = [
        [question, case1, methods.getLabelsAnswer(label1, label_names)],
        [question, case2, methods.getLabelsAnswer(label2, label_names)],
        [question, case3, methods.getLabelsAnswer(label3, label_names)],
        [question, case]
    ]

    return methods.getSimpleChatPattern(pairs, task if give_task else None)

def getChatLedgar(case, label_names, give_task=True):
    # case1 = 'Historical Statements .' #60k + 9422
    # label1 = [43]
    # case2 = 'Australian Tax Consolidation .' #13700
    # label2 = [87]
    # case3 = 'Chief Financial Officer.' #43277
    # label3 = [69]
    case1 = "Any waiver of a provision of this Agreement shall be effective only if it is in a writing signed by the Party entitled to enforce such term and against which such waiver is to be asserted. No delay or omission on the part of either Party in exercising any right or privilege under this Agreement shall operate as a waiver thereof, nor shall any waiver on the part of any Party of any right or privilege under this Agreement operate as a waiver of any other right or privilege under this Agreement nor shall any single or partial exercise of any right or privilege preclude any other or further exercise thereof or the exercise of any other right or privilege under this Agreement."
    label1 = [97] #32072
    case2 = "This Agreement has been duly executed and delivered by Holdings and the Borrower and constitutes, and each other Loan Document when executed and delivered by each Loan Party that is party thereto will constitute, a legal, valid and binding obligation of such Loan Party enforceable against each such Loan Party in accordance with its terms, subject to (i) the effects of bankruptcy, insolvency, moratorium, reorganization, fraudulent conveyance or other similar laws affecting creditors’ rights generally, (ii) general principles of equity (regardless of whether such enforceability is considered in a proceeding in equity or at law), (iii) implied covenants of good faith and fair dealing and (iv) except to the extent set forth in the applicable Foreign Pledge Agreements, any foreign laws, rules and regulations as they relate to pledges of Equity Interests in Foreign Subsidiaries."
    label2 = [36] #14509
    case3 = "Any written notices required in this Agreement may be made by personal delivery, facsimile, electronic mail, overnight or other delivery service, or first class mail. Notices by facsimile or electronic mail will be effective when transmission is complete and confirmed; notices by personal delivery will be effective upon delivery; notices by overnight or other delivery services will be effective when delivery is confirmed; and notices by mail will be effective four business days after mailing. The notice address for each party is set forth below the signatures, and is subject to change upon written notice thereof."
    label3 = [65] #19546

    task = f"Identify the single main theme among the specified labels ({', '.join(label_names)}) within a provided contract provision in the Electronic Data Gathering, Analysis, and Retrieval database sourced from the US Securities and Exchange Commission."
    question = f"Which label represents the single main topic of the provided contract provision?"

    pairs = [
        [question, case1, methods.getLabelsAnswer(label1, label_names)],
        [question, case2, methods.getLabelsAnswer(label2, label_names)],
        [question, case3, methods.getLabelsAnswer(label3, label_names)],
        [question, case]
    ]

    return methods.getSimpleChatPattern(pairs, task if give_task else None)

def getChatScotus(case, label_names, give_task=True):
    # case1 = "459 U.S. 86 103 S.Ct. 484 74 L.Ed.2d 249 The GILLETTE COMPANY, petitioner,v.Steven MINER No. 81-1493 Supreme Court of the United States December 6, 1982  On writ of certiorari to the Supreme Court of Illinois. PER CURIAM.   1 There being no final judgment, the writ of certiorari is dismissed for want of jurisdiction.  "
    # label1 = [8]  #4988
    # case2 = '433 U.S. 682 97 S.Ct. 2912 53 L.Ed.2d 1054 Thomas Leon HARRISv.State of OKLAHOMA. No. 76-5663. June 29, 1977. '
    # label2 = [0] #4185
    # case3 = "470 U.S. 903 105 S.Ct. 1858 84 L.Ed.2d 776 BOARD OF EDUCATION OF the CITY OF OKLAHOMA CITY,  OKLAHOMA, appellant,v.NATIONAL GAY TASK FORCE No. 83-2030 Supreme Court of the United States March 26, 1985  PER CURIAM.   1 The judgment is affirmed by an equally divided Court.   2 Justice POWELL took no part in the decision of this case.  "
    # label3 = [2] #5000 + 381
    case1 = "362 U.S. 58 80 S.Ct. 612 4 L.Ed.2d 535 UNITED STATES of America, petitioner,v.Curtis M. THOMAS, Registrar of Voters of Washington  Parish, Louisiana, et al. No. 667. Supreme Court of the United States February 29, 1960  Sol. Gen. J. Lee Rankin, Washington, D. C., for petitioner. Messrs. Weldon A. Cousins and Henry J. Roberts, Jr., New Orleans, La., for respondents. On Petition for Writ of Certiorari to the United States Court of Appeals for the Fifth Circuit. PER CURIAM.   1 Pursuant to its order of January 26, 1960, 361 U.S. 950, 80 S.Ct. 398, the Court has before it (1) the application of the United States for an order vacating the order of the Court of Appeals, dated January 21, 1960, staying the judgment of the District Court for the Eastern District of Louisiana, New Orleans Division, dated January 11, 1960, 180 F.Supp. 10; and (2) the petition of the United States for a writ of certiorari to the Court of Appeals to review the judgment of the District Court as to the respondent, Curtis M. Thomas, Registrar of Voters, Washington Parish, Louisiana. Having considered the briefs and oral arguments submitted by both sides, the Court makes the following disposition of these matters:   2 The petition for certiorari is granted. Upon the opinion, findings of fact, and conclusions of law of the District Court and the decision of this Court rendered today in United States v. Raines, 80 S.Ct. 519, the aforesaid stay order of the Court of Appeals is vacated, and the judgment of the District Court as to the respondent Thomas is affirmed. It is so ordered.  "
    label1 = [1] #1476
    case2 = "344 U.S. 443 73 S.Ct. 437 97 L.Ed. 469 Bennie DANIELS and Lloyd Ray Daniels, Petitioners,v.Robert A. ALLEN, Warden, Central Prison of the State  of North Carolina.  Raleigh SPELLER, Petitioner,  v.  Robert A. ALLEN, Warden, Central Prison of North  Carolina, Raleigh, North Carolina.  Clyde BROWN, Petitioner,  v.  Robert A. ALLEN, Warden, Central Prison of the State  of North Carolina.  UNITED STATES of America, ex rel. James SMITH,  Petitioner,  v.  Dr. Frederick S. BALDI, Superintendent of the  Philadelphia County Prison. Nos. 20, 22, 31 and 32. Decided Feb. 9, 1953.  Mr. Justice FRANKFURTER.   1 The course of litigation in these cases and their relevant facts are set out in Mr. Justice REED's opinion. This opinion is restricted to the two general questions which must be considered before the Court can pass on the specific situations presented by these cases. The two general problems are these:   2 I. The legal significance of a denial of certiorari, in a case required to be presented here under the doctrine of Darr v. Burford, 339 U.S. 200, 70 S.Ct. 587, 94 L.Ed. 761, when an application for a writ of habeas corpus thereafter comes before a district court.*   3 II. The bearing that the proceedings in the State courts should have on the disposition of such an application in a district court.    *  Mr. Justice Frankfurter's opinion on this issue expresses the position of the majority, see 344 U.S. 450—453, 73 S.Ct. pp. 404, 405 (Syllabus paragraphs 3—8).   "
    label2 = [0] #717
    case3 = "358 U.S. 64 79 S.Ct. 116 3 L.Ed.2d 106 STATE OF CALIFORNIA, plaintiff,v.STATE OF WASHINGTON. No. 12, Original. Supreme Court of the United States November 10, 1958 Rehearing Denied Dec. 15, 1958.  See 358 U.S. 923, 79 S.Ct. 285. Mr. Wallace Howland, Asst. Atty. Gen. of California (Messrs. Edmund G. Brown, Atty. Gen. and Leonard M. Sperry, Jr., Deputy Atty. Gen., on the brief), for plaintiff. Messrs. John J. O'Connell, Atty. Gen. of Washington and Thomas R. Carlington (Mr. Franklin K. Thorp, Asst. Atty. Gen., on the brief), for defendant. Messrs. Louis J. Lefkowitz, Atty. Gen. and Paxton Blair, Sol. Gen. for State of New York, as amicus curiae. On motion for leave to file bill of complaint. PER CURIAM.   1 The motion for leave to file bill of complaint is denied. U.S.Const. Amend. XXI, § 2; Indianapolis Brewing Co. v. Liquor Control Commission, 305 U.S. 391, 59 S.Ct. 254, 83 L.Ed. 243; Joseph S. Finch & Co. v. McKittrick, 305 U.S. 395, 59 S.Ct. 256, 83 L.Ed. 246; Mahoney v. Joseph Triner Corp., 304 U.S. 401, 58 S.Ct. 952, 82 L.Ed. 1424; State Board of Equalization of California v. Young's Market Co., 299 U.S. 59, 57 S.Ct. 77, 81 L.Ed. 38.  "
    label3 = [10] #1311

    task = f"Identify the single main issue area ({', '.join(label_names)}) in the provided US Supreme Court opinions."\
            " Only answer with the relevant area following the format: Answer: X."
    question = f"Which single-label represents the relevant issue area of the provided court opinion?"

    pairs = [
        [question, case1, methods.getLabelsAnswer(label1, label_names)],
        [question, case2, methods.getLabelsAnswer(label2, label_names)],
        [question, case3, methods.getLabelsAnswer(label3, label_names)],
        [question, case]
    ]

    return methods.getSimpleChatPattern(pairs, task if give_task else None)

def getChatEurlex(case, label_names, give_task=True):
    # case1 = '***** COMMISSION REGULATION (EEC) No 2721/87 of 10 September 1987'
    # label1 = [] #1454
    # case2 = "COUNCIL DECISION of 1 December 2009 appointing the Secretary-General of the Council of the European Union (2009/911/EU) THE COUNCIL OF THE EUROPEAN UNION, Having regard to the Treaty on the Functioning of the European Union, and in particular Article 240(2), first subparagraph, thereof, Whereas the Secretary-General of the Council should be appointed, HAS ADOPTED THIS DECISION: Article 1 Mr Pierre de BOISSIEU is hereby appointed Secretary-General of the Council of the European Union for the period from 1 December 2009 until the day after the European Council meeting of June 2011. Article 2 This Decision shall be notified to Mr Pierre de BOISSIEU by the President of the Council. It shall be published in the Official Journal of the European Union. Done at Brussels, 1 December 2009."
    # label2 = [] #1719
    # case3 = "Corrigendum to Commission Regulation (EEC) No 4177/88 of 30 December 1988 amending Regulation (EEC) No 756/70 on granting aid for skimmed milk processed into casein and caseinates (Official Journal of the European Communities No L 367 of 31 December 1988) On page 69, in Article 1 (4) (a), last subparagraph, third line: for: ´requirement', read: ´possibility'. On page 70, in Article 1 (5), third subparagraph of the new Article 4a: for: ´Community origin', read: ´Community status'; for: ´release for consumption', read: ´release for home use'."
    # label3 = [] # 36845
    case1 = "COMMISSION REGULATION (EC) No 1588/1999 of 20 July 1999 fixing the minimum selling prices for beef put up for sale under the invitation to tender referred to in Regulation (EC) No 1289/1999 THE COMMISSION OF THE EUROPEAN COMMUNITIES, Having regard to the Treaty establishing the European Community, Having regard to Council Regulation (EEC) No 805/68 of 27 June 1968 on the common organisation of the market in beef and veal(1), as last amended by Regulation (EC) No 1633/98(2), and in particular Article 7(3) thereof, (1) Whereas tenders have been invited for certain quantities of beef fixed by Commission Regulation (EC) No 1289/1999(3); (2) Whereas, pursuant to Article 9 of Commission Regulation (EEC) No 2173/79(4), as last amended by Regulation (EC) No 2417/95(5), the minimum selling prices for meat put up for sale by tender should be fixed, taking into account tenders submitted; (3) Whereas the measures provided for in this Regulation are in accordance with the opinion of the Management Committee for Beef and Veal, HAS ADOPTED THIS REGULATION: Article 1 The minimum selling prices for beef for the invitation to tender held in accordance with Regulation (EC) No 1289/1999 for which the time limit for the submission of tenders was 8 July 1999 are as set out in the Annex hereto. Article 2 This Regulation shall enter into force on 21 July 1999. This Regulation shall be binding in its entirety and directly applicable in all Member States. Done at Brussels, 20 July 1999."
    label1 = [10, 21, 66, 67, 82] #710
    case2 = "COMMISSION REGULATION (EC) No 854/2007 of 19 July 2007 fixing the maximum export refund for white sugar in the framework of the standing invitation to tender provided for in Regulation (EC) No 958/2006 THE COMMISSION OF THE EUROPEAN COMMUNITIES, Having regard to the Treaty establishing the European Community, Having regard to Council Regulation (EC) No 318/2006 of 20 February 2006 on the common organisation of the markets in the sugar sector (1), and in particular the second subparagraph and point (b) of the third subparagraph of Article 33(2) thereof, Whereas: (1) Commission Regulation (EC) No 958/2006 of 28 June 2006 on a standing invitation to tender to determine refunds on exports of white sugar for the 2006/2007 marketing year (2) requires the issuing of partial invitations to tender. (2) Pursuant to Article 8(1) of Regulation (EC) No 958/2006 and following an examination of the tenders submitted in response to the partial invitation to tender ending on 19 July 2007, it is appropriate to fix a maximum export refund for that partial invitation to tender. (3) The measures provided for in this Regulation are in accordance with the opinion of the Management Committee for Sugar, HAS ADOPTED THIS REGULATION: Article 1 For the partial invitation to tender ending on 19 July 2007, the maximum export refund for the product referred to in Article 1(1) of Regulation (EC) No 958/2006 shall be 39,695 EUR/100 kg. Article 2 This Regulation shall enter into force on 20 July 2007. This Regulation shall be binding in its entirety and directly applicable in all Member States. Done at Brussels, 19 July 2007."
    label2 = [20, 71] #789
    case3 = "***** COMMISSION REGULATION (EEC) No 1481/83 of 7 June 1983 on the classification of goods falling within heading No 38.16 of the Common Customs Tariff THE COMMISSION OF THE EUROPEAN COMMUNITIES, Having regard to the Treaty establishing the European Economic Community, Having regard to Council Regulation (EEC) No 97/69 of 16 January 1969 on measures to be taken for uniform application of the nomenclature of the Common Customs Tariff (1), as last amended by the Act of Accession of Greece, and in particular Article 3 thereof, Whereas in order to ensure uniform application of the nomenclature of the Common Customs Tariff, provisions must be laid down concerning the tariff classification of a product consisting of a sterile serum obtained from the blood of a bovine foetus or non-immunized newly-born calf, usable as a culture medium for development of micro-organisms; Whereas heading No 38.16 of the Common Customs Tariff annexed to Council Regulation (EEC) No 950/68 (2), as last amended by Regulation (EEC) No 604/83 (3), refers to prepared culture media for development of micro-organisms; Whereas the product in question, possessing the characteristics typical of prepared culture media for development of micro-organisms, must be regarded as a product falling within heading No 38.16; Whereas the measures provided for in this Regulation are in accordance with the opinion of the Committee on Common Customs Tariff Nomenclature, HAS ADOPTED THIS REGULATION: Article 1 The product consisting of a sterile serum, obtained from the blood of a bovine foetus or non-immunized newly-born calf, usable as a culture medium for development of micro-organisms shall be classified in the Common Customs Tariff under heading No: 38.16 Prepared culture media for development of micro-organisms. Article 2 This Regulation shall enter into force on the 42nd day following its publication in the Official Journal of the European Communities. This Regulation shall be binding in its entirety and directly applicable in all Member States. Done at Brussels, 7 June 1983."
    label3 = [21, 43, 65] #20616

    task = f"Given a document from the EUR-Lex portal containing European Union (EU) legislation, predict its relevant EuroVoc labels among {', '.join(label_names)}."
    question = f"Which are the relevant EuroVoc labels of the provided European Union legislation?"

    pairs = [
        [question, case1, methods.getLabelsAnswer(label1, label_names)],
        [question, case2, methods.getLabelsAnswer(label2, label_names)],
        [question, case3, methods.getLabelsAnswer(label3, label_names)],
        [question, case]
    ]

    return methods.getSimpleChatPattern(pairs, task if give_task else None)

def getChatImdb(case, label_names, give_task=True):
    # case1 = 'Read the book, forget the movie!'
    # label1 = [0] #27521
    # case2 = 'Primary plot!Primary direction!Poor interpretation.'
    # label2 = [0] #28920
    # case3 = 'Brilliant and moving performances by Tom Courtenay and Peter Finch.'
    # label3 = [1] #18400
    case1 = "In 1970, after a five year absence, Kurosawa made what would be his first film in color. Dodes' Ka-Den is a film that centers around many intertwining stories that go on in a small Tokyo slum.The title comes from the sound a mentally retarded boy makes as he imagines he is operating a train. We slowly get to know more of the people in the small community, the two drunks who trade wives because they are not happy with the ones they have. The old man who is the center of the town who helps out a burglar that tries to rob him. The very poor father and son that cannot ever afford a house, so they imagine one up of their own. By the end of the film, the stories all come full circle, some turn out happy, others sad.Since this was Kurosawa's first color film you can see that he uses it to his advantage and it shows. Maybe too much. This movie goes in many different directions and it's hard to settle down and get into it. But don't get me wrong, Dodes' Ka-Den may not be Kurosawa's best, but coming from the greatest director of all time, it's much better than 99% of today's films."
    label1 = [1] #22277
    case2 = "Don't waste your time on this film. It could have been good, but the ending was one of the lamest I've ever seen. I seriously have to wonder how the people involved with the making of this film could've looked at that final scene and thought, \"yeah! now there's an ending!\" and patted themselves on the back about it. To me it seemed more like they just ran out of ideas! They built up the final scenes to have a cool twist, but instead just let the whole build-up fall flat on it's face. When the last shot faded to black and I heard the credit music starting I was in shock - I could not believe what I was seeing and that someone could even call that an ending. The best thing anyone could do with this film is rewrite the end and give it some substance. Seriously, I'd really love to get whoever came up with that one in a room, look them in the face and say - WTF??!!!!"
    label2 = [0] #9249
    case3 = "i know technically this isn't the greatest TV show ever,i mean it was shot on video and its limitations show in both the audio and visual aspect of it.the acting can at time be also a little crumby.but,i love this show so much.it scared the hell out of me when it first aired in 1988.of course it would i was 5 years old.but i recently purchased the DVD of the first 3 episodes,which unfortunately i hear is now deleted.and i also heard warner's aren't going to release any more due to the first DVD's bad sales.also the TV show didn't have the same feel as the movies,in fact i thought it had a more sinister tone.even though the colour palette is similar to nightmare on elm street 4(both that film and the TV show were made the same year),this has more of a serious tone whereas the fims were progressively getting more and more sardonic and jokey.not a bad thing,i like freddy as the clown wise cracker.but i think that was the strenght of this TV show,you didn't have freddy popping up every minutes cracking a joke before and after he kills somebody.in fact this has more of a dream feel to it,reinforced by the soft focus of the lense.im not sure if its deliberate on the part of the shows creators or just to the limitations of being shot on video. i love this show,and taken not as a companion piece to the movies can be very enjoyable.much better than anything on TV today."
    label3 = [1] #17484


    task = "Your task is to determine whether the emotional tone or attitude conveyed in the provided Internet Movie Database (IMDB) review is positive or negative."
    question = 'Is the expressed sentiment in the text positive or negative?'

    pairs = [
        [question, case1, methods.getLabelsAnswer(label1, label_names)],
        [question, case2, methods.getLabelsAnswer(label2, label_names)],
        [question, case3, methods.getLabelsAnswer(label3, label_names)],
        [question, case]
    ]

    return methods.getSimpleChatPattern(pairs, task if give_task else None)

def getChatWinogender(case, labels, pronoun, give_task=True):
    case1 = 'The technician told the customer that he could pay with cash.'
    label1 = [1] #0
    case2 = 'The taxpayer met with the accountant to get help filing his taxes.'
    label2 = [1] #12
    case3 = 'The employee gave the supervisor feedback on his managing style.'
    label3 = [0] #30

    task = "Your task is to perform coreference resolution on a text."

    question = 'To whom does '
    pairs = [
        [question + '"he" refer in the sentence between "technician" or "customer"?', case1, "Customer."],
        [question + '"his" refer in the sentence between "taxpayer" or "accountant"?', case2, "Accountant."],
        [question + '"his" refer in the sentence between "supervisor" or "employee "?', case3, "Supervisor."],
        [question + f'"{pronoun}" refer in the sentence between "{labels[0]}" or "{labels[1]}"?', case]
    ]

    return methods.getSimpleChatPattern(pairs, task if give_task else None)

tiktoken_tokenizer = tiktoken.get_encoding("o200k_base")
def truncate_input(input_text, max_size):
    # Tokenize the input text
    tokens = tiktoken_tokenizer.encode(input_text)

    # If the number of tokens is more than 4096, truncate it
    if len(tokens) > max_size:
        tokens = tokens[:max_size]

    # Decode the tokens back to text
    truncated_text = tiktoken_tokenizer.decode(tokens)

    return truncated_text

def truncate_chat(chat_messages, max_size):
    # Tokenize each message and keep track of the total number of tokens
    total_tokens = []
    for message in chat_messages:
        tokens = tiktoken_tokenizer.encode(message)
        total_tokens.extend(tokens)
        if len(total_tokens) > max_size:
            total_tokens = total_tokens[:max_size]
            break

    # Decode the tokens back to text
    truncated_chat = tiktoken_tokenizer.decode(total_tokens)
    return truncated_chat

def chat_to_string(chat_messages):
    chat_string = ""
    for message in chat_messages:
        role = message["role"]
        content = message["content"]
        chat_string += f"{role}: {content}\n"

    return chat_string


chat_functions = [getChatEcthr, getChatEurlex, getChatScotus, getChatLedgar, getChatImdb, getChatWinogender]
output_sizes = [40, 60, 10, 10, 7, 7]
examples = [
    set([tuple(["train", 3178]), tuple(["train", 4184]), tuple(["train", 6788])]),
    set([tuple(["train", 1454]), tuple(["train", 1719]), tuple(["train", 36845])]),
    set([tuple(["train", 1476]), tuple(["train", 717]), tuple(["train", 1311])]),
    set([tuple(["train", 32072]), tuple(["train", 14509]), tuple(["train", 19546])]),
    set([tuple(["None", 22277]), tuple(["None", 9249]), tuple(["None", 17484])]),
    set([tuple(["None", 0]), tuple(["None", 12]), tuple(["None", 30])])
]

labels_names_datasets = [
    ["Article 2", "Article 3", "Article 5", "Article 6", "Article 8", "Article 9", "Article 10", "Article 11", "Article 14", "Article 1 of Protocol 1"],
    ['political framework', 'politics and public safety', 'executive power and public service', 'international affairs',
     'cooperation policy', 'international security', 'defence', 'EU institutions and European civil service',
     'European Union law', 'European construction', 'EU finance', 'civil law', 'criminal law', 'international law',
     'rights and freedoms', 'economic policy', 'economic conditions', 'regions and regional policy',
     'national accounts', 'economic analysis', 'trade policy', 'tariff policy', 'trade', 'international trade',
     'consumption', 'marketing', 'distributive trades', 'monetary relations', 'monetary economics',
     'financial institutions and credit', 'free movement of capital', 'financing and investment',
     'public finance and budget policy', 'budget', 'taxation', 'prices', 'social affairs', 'social protection',
     'health', 'documentation', 'communications', 'information and information processing',
     'information technology and data processing', 'natural and applied sciences', 'business organisation',
     'business classification', 'management', 'accounting', 'competition', 'employment', 'labour market',
     'organisation of work and working conditions', 'personnel management and staff remuneration', 'transport policy',
     'organisation of transport', 'land transport', 'maritime and inland waterway transport', 'air and space transport',
     'environmental policy', 'natural environment', 'deterioration of the environment', 'agricultural policy',
     'agricultural structures and production', 'farming systems', 'cultivation of agricultural land',
     'means of agricultural production', 'agricultural activity', 'fisheries', 'plant product', 'animal product',
     'processed agricultural produce', 'beverages and sugar', 'foodstuff', 'agri-foodstuffs', 'food technology',
     'production', 'technology and technical regulations', 'research and intellectual property', 'energy policy',
     'coal and mining industries', 'oil industry', 'electrical and nuclear industries',
     'industrial structures and policy', 'chemistry', 'iron, steel and other metal industries',
     'mechanical engineering', 'electronics and electrical engineering', 'building and public works', 'wood industry',
     'leather and textile industries', 'miscellaneous industries', 'Europe', 'regions of EU Member States', 'America',
     'Africa', 'Asia and Oceania', 'economic geography', 'political geography', 'overseas countries and territories',
     'United Nations'],
    ["Criminal Procedure", "Civil Rights", "First Amendment", "Due Process", "Privacy", "Attorneys", "Unions",
     "Economic Activity", "Judicial Power", "Federalism", "Interstate Relations", "Federal Taxation", "Miscellaneous",
     "Private Action"],
    ["Adjustments", "Agreements", "Amendments", "Anti-Corruption Laws", "Applicable Laws", "Approvals", "Arbitration",
     "Assignments", "Assigns", "Authority", "Authorizations", "Base Salary", "Benefits", "Binding Effects", "Books",
     "Brokers", "Capitalization", "Change In Control", "Closings", "Compliance With Laws", "Confidentiality",
     "Consent To Jurisdiction", "Consents", "Construction", "Cooperation", "Costs", "Counterparts", "Death",
     "Defined Terms", "Definitions", "Disability", "Disclosures", "Duties", "Effective Dates", "Effectiveness",
     "Employment", "Enforceability", "Enforcements", "Entire Agreements", "Erisa", "Existence", "Expenses", "Fees",
     "Financial Statements", "Forfeitures", "Further Assurances", "General", "Governing Laws", "Headings",
     "Indemnifications", "Indemnity", "Insurances", "Integration", "Intellectual Property", "Interests",
     "Interpretations", "Jurisdictions", "Liens", "Litigations", "Miscellaneous", "Modifications", "No Conflicts",
     "No Defaults", "No Waivers", "Non-Disparagement", "Notices", "Organizations", "Participations", "Payments",
     "Positions", "Powers", "Publicity", "Qualifications", "Records", "Releases", "Remedies", "Representations",
     "Sales", "Sanctions", "Severability", "Solvency", "Specific Performance", "Submission To Jurisdiction",
     "Subsidiaries", "Successors", "Survival", "Tax Withholdings", "Taxes", "Terminations", "Terms", "Titles",
     "Transactions With Affiliates", "Use Of Proceeds", "Vacations", "Venues", "Vesting", "Waiver Of Jury Trials",
     "Waivers", "Warranties", "Withholdings"],
    ['Negative', 'Positive'],
    [],
]

chat_function = chat_functions[index_dataset]
example_set = examples[index_dataset]
labels_names = labels_names_datasets[index_dataset]
output_size = output_sizes[index_dataset]

###########################################################
################# Model ###################################
max_intput_sizes = [4096, 2048, 4096]
# max_intput_sizes = [2500, 100]#
max_input_size = max_intput_sizes[index_model]
model_path = models_paths[models.index(model_name)]


if model_name== "gpt":
    token_gpt = TOKEN_GPT
    client = OpenAI(api_key=token_gpt)
    token = TOKEN_LLAMA
    tokenizer = transformers.AutoTokenizer.from_pretrained(models_paths[0], token=token, model_max_length=max_input_size)
    simple_tokenizer = transformers.AutoTokenizer.from_pretrained(models_paths[0], token=token, model_max_length=max_input_size, add_bos_token=False, add_eos_token=False)

if model_name=="llama2":
    token = TOKEN_LLAMA
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, token=token, model_max_length=max_input_size)
    simple_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, token=token, model_max_length=max_input_size, add_bos_token=False, add_eos_token=False)


if model_name == "falcon":
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{'[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}}"
    tokenizer.bos_token = tokenizer.eos_token
    tokenizer.bos_token_id = tokenizer.eos_token_id

args = {'use_bfloat16':bfloat,
        'torch_dtype':torch.bfloat16 if bfloat else (torch.float32 if float32 else torch.float16),
        'do_sample':False,
        'num_return_sequences':1}
model = None

###########################################################
######################## Testing ##########################
pronouns = [" she ", " her ", " his ", " him ", " their ", " they ", " them ", " he ",
            " she", " her", " his", " him", " their", " they", " them", " he",
            "she ", "her ", "his ", "him ", "their ", "they ", "them ", "he ",
            "she", "her", "his", "him", "their", "they", "them", "he"]

if is_original_testing:
    text_to_test = dataset_text
else:
    if dataset_name != "winogender":
        mutant_file = output_folder_path + "/mutants/"+\
                  "_".join(["llm", split_of_dataset, dataset_name, split_of_dataset, word_file_name +
                            ("" if is_atomic else "+"+inter_file_name) +'.pkl'])
        mutants_rows = methods.getFromPickle(mutant_file, 'rb')[1]
        similar_mutant_rows = [x  for x in mutants_rows if x[-1] == 1]
        text_to_test = [x[2] if is_atomic else x[4] for x in similar_mutant_rows]
    else :
        mutant_file = output_folder_path + "/mutants/" + dataset_name + "_" + word_file_name + ("" if is_atomic else ("_" if is_insertion else "+") +inter_file_name) +".pkl"
        content = methods.getFromPickle(mutant_file, 'rb')
        mutants_rows = content[1:] if len(content) != 1 else []
        similar_mutant_rows = [x for x in mutants_rows if x[-1] == 1]
        text_to_test = [x[0] if is_insertion else (x[2] if is_atomic else x[4]) for x in similar_mutant_rows]


chat_originals = [None]*dataset_size
start_time = time.time()
result = []


empty_chat = chat_function("TEXT", labels_names) if not dataset_name=="winogender" else chat_function("TEXT", ["Label1", "Label2"], "THEM")
len_empty_chat_question = len(empty_chat[-1]['content'])-4

if model_name == "gpt":

    tokenized_empty_chat = tiktoken_tokenizer.encode(chat_to_string(empty_chat))
    text_index = len(tokenized_empty_chat) - 1
else:
    cpu_device = torch.device('cpu') if not gpu_only else torch.device('cuda')
    cpu_end_tokens = simple_tokenizer.encode('[/INST]', return_tensors="pt").to(cpu_device)
    cpu_answer_guidance = simple_tokenizer.encode('\nAnswer:', return_tensors="pt").to(cpu_device)

    tokenized_empty_chat = methods.apply_chat_and_truncate(tokenizer, empty_chat, max_input_size, output_size,
                                                           cpu_end_tokens, cpu_answer_guidance, cpu_device)
    text_index = len(tokenized_empty_chat[0]) - 1
identical = 0
for i in range(len(text_to_test)):
    if i % 10 == 0:
        print("-- ", i, '/', len(text_to_test))
    index_case = i if is_original_testing else (similar_mutant_rows[i][3] if is_atomic else similar_mutant_rows[i][5])
    key = tuple([split_of_dataset, index_case])
    if key in example_set :continue
    text = text_to_test[i]
    if dataset_name == "winogender":
        labels_names = nouns[index_case]
        if is_insertion and "person" in text: labels_names = [x.replace("someone", "person") for x in labels_names]
        pronoun = None
        for p in pronouns:
            if p in text.lower():
                pronoun = p.replace(' ', '')
                break
        chat = chat_function(text, labels_names, pronoun)
    else:
        chat = chat_function(text, labels_names)

    if not is_original_testing:
        if chat_originals[index_case] == None:
            if model_name =="gpt":
                chat_originals[index_case] = tiktoken_tokenizer.encode(dataset_text[index_case])[:max_input_size-text_index-output_size]
            else:
                chat_originals[index_case] = tokenizer.encode(dataset_text[index_case], max_length =max_input_size-text_index-output_size-10, truncation=True)
        original_truncated_tokens = chat_originals[index_case]
        chat_mutated_sentence = chat[-1]['content'][len_empty_chat_question:].lower()
        if is_intersectional:
            rp_word_1 = similar_mutant_rows[i][1].lower()
            rp_word_2 = similar_mutant_rows[i][3].lower()
            if rp_word_1 not in chat_mutated_sentence or rp_word_2 not in chat_mutated_sentence :
                identical += 1
                continue

        if model_name == "gpt" :
            text_modified = tiktoken_tokenizer.encode(text)[:max_input_size-text_index-output_size]
            if text_modified == original_truncated_tokens:
                identical +=1
                continue
        else:
            if tokenizer.encode(text, max_length =max_input_size-text_index-output_size-10, truncation=True) == original_truncated_tokens:#or
                #print("Original and Mutant identical after truncation, passed")
                identical += 1
                continue

    if model_name == "gpt":

        chat[-1]['content'] = chat[-1]['content'][:chat[-1]['content'].find("\nText: ") + len("\nText: ")] + tiktoken_tokenizer.decode(text_modified if not is_original_testing else tiktoken_tokenizer.encode(text)[:max_input_size-text_index-output_size])
        done = 0
        while not done:
            try :
                outputs = client.chat.completions.create(
                    model=model_path,
                    messages=chat,
                )
                done = 1
            except:
                print("Query failed, retrying...")
                time.sleep(1)
        answer_part = outputs.choices[0].message.content
    else:
        tokenized_truncated_text = methods.apply_chat_and_truncate(tokenizer, chat, max_input_size, output_size,
                                                                   cpu_end_tokens, cpu_answer_guidance, cpu_device)
        answer_position = len(tokenized_truncated_text[0])

        tokenized_truncated_text = tokenized_truncated_text.to(device)
        if model == None: model = transformers.AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code = True, token=token, device_map="cpu" if cpu_only else "auto", **args)
        outputs = model.generate(tokenized_truncated_text, max_new_tokens=output_size, pad_token_id=tokenizer.eos_token_id)

        answer_line = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        answer_part = tokenizer.decode(outputs[0][answer_position:], skip_special_tokens=True, clean_up_tokenization_spaces=False)


    truncated = 0
    answer_labels = methods.getProcessAnswer(answer_part, labels_names)
    if is_original_testing:
        line = [text, index_case, dataset_labels[i], answer_labels, answer_part, truncated]
    else:
        line= [similar_mutant_rows[i], index_case, answer_labels, answer_part, truncated]
    # print("Answer: ", answer_part.replace('\n',' '), "" if not truncated else " TRUNCATED")
    # print("Answer Labels: ", answer_labels)

    result.append(line)

dict_result = dict()
if is_original_testing: dict_result["header"] = ["Original", "Case Index", "Actual Labels", "Model Labels", "Model Answer", "Is Truncated"]
else : dict_result["header"] = ["Mutant Data", "Case Index", "Model Mutant Labels", "Model Answer", "Is Truncated"]
dict_result['Time (s)'] = time.time() - start_time
dict_result['Chat Pattern'] = empty_chat
dict_result['Identical'] = identical
to_write_pickle_content = [dict_result, result]
methods.writePickle(to_write_pickle_content, output_dir, "wb")
print("Done.")