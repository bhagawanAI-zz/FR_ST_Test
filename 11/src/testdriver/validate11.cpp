/**
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#include <fstream>
#include <iostream>
#include <cstring>
#include <iterator>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>


#include "frvt11.h"
#include "util.h"

using namespace std;
using namespace FRVT;
using namespace FRVT_11;

int
readTemplateFromFile(
        const string &filename,
        vector<uint8_t> &templ)
{
    streampos fileSize;
    ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << filename << "." << endl;
        return FAILURE;
    }
    file.seekg(0, ios::end);
    fileSize = file.tellg();
    file.seekg(0, ios::beg);

    templ.resize(fileSize);
    file.read((char*)&templ[0], fileSize);
    return SUCCESS;
}

int
createTemplate(
        std::shared_ptr<Interface> &implPtr,
        const string &inputFile,
        const string &outputLog,
        const string &templatesDir,
        TemplateRole role)
{
    /* Read input file */
    ifstream inputStream(inputFile);
    if (!inputStream.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << inputFile << "." << endl;
        raise(SIGTERM);
    }

    /* Open output log for writing */
    ofstream logStream(outputLog);
    if (!logStream.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << outputLog << "." << endl;
        raise(SIGTERM);
    }

    /* header */
    logStream << "id image templateSizeBytes returnCode isLeftEyeAssigned "
            "isRightEyeAssigned xleft yleft xright yright" << endl;

    string id, line; 
    while (std::getline(inputStream, line)) {
        auto tokens = split(line, ' ');
        id = tokens[0];
        // Get number of image entries in line
        auto numImages = (tokens.size() - 1)/2;

        Multiface faces;
        for (unsigned int i=0; i<numImages; i++) {
            Image image;
            string imagePath = tokens[(i*2)+1];
            string desc = tokens[(i*2)+2];
            if (!readImage(imagePath, image)) {
                cerr << "Failed to load image file: " << imagePath << "." << endl;
                raise(SIGTERM);
            }
            image.description = mapStringToImgLabel[desc];
            faces.push_back(image);
        }

        vector<uint8_t> templ;
        vector<EyePair> eyes;
        auto ret = implPtr->createTemplate(faces, role, templ, eyes);
        
        /* Check that function is implemented */
        if (ret.code == ReturnCode::NotImplemented) {
            cerr << "[ERROR] The createTemplate(faces, role, templ, eyes) function returned ReturnCode::NotImplemented.  This function must be implemented!" << std::endl;
            raise(SIGTERM);
        }

        /* Open template file for writing */
        string templFile{id + ".template"};
        ofstream templStream(templatesDir + "/" + templFile);
        if (!templStream.is_open()) {
            cerr << "[ERROR] Failed to open stream for " << templatesDir + "/" + templFile << "." << endl;
            raise(SIGTERM);
        }

        /* Write template file */
        templStream.write((char*)templ.data(), templ.size());

        /* Write template stats to log */
        for (unsigned int i = 0; i< faces.size(); i++) {
            string imagePath = tokens[(i*2)+1];
            logStream << id << " "
                << imagePath << " "
                << templ.size() << " "
                << static_cast<std::underlying_type<ReturnCode>::type>(ret.code) << " "
                << (eyes.size() > 0 ? eyes[i].isLeftAssigned : false) << " "
                << (eyes.size() > 0 ? eyes[i].isRightAssigned : false) << " "
                << (eyes.size() > 0 ? eyes[i].xleft : 0) << " "
                << (eyes.size() > 0 ? eyes[i].yleft : 0) << " "
                << (eyes.size() > 0 ? eyes[i].xright : 0) << " "
                << (eyes.size() > 0 ? eyes[i].yright : 0)
                << endl;
        }
    }
    inputStream.close();

    /* Remove the input file */
    if( remove(inputFile.c_str()) != 0 )
        cerr << "Error deleting file: " << inputFile << endl;

    return SUCCESS;
}

int
createMultiTemplates(
        std::shared_ptr<Interface> &implPtr,
        const string &inputFile,
        const string &outputLog,
        const string &templatesDir,
        TemplateRole role)
{
    /* Read input file */
    ifstream inputStream(inputFile);
    if (!inputStream.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << inputFile << "." << endl;
        raise(SIGTERM);
    }

    /* Open output log for writing */
    ofstream logStream(outputLog);
    if (!logStream.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << outputLog << "." << endl;
        raise(SIGTERM);
    }

    /* header */
    logStream << "id image templateSizeBytes returnCode numDetections detectionIndex isLeftEyeAssigned "
            "isRightEyeAssigned xleft yleft xright yright" << endl;

    string id, line; 
    while (std::getline(inputStream, line)) {
        auto tokens = split(line, ' ');
        id = tokens[0];
        // Get number of image entries in line
        auto numImages = (tokens.size() - 1)/2;
        if (numImages != 1) {
            cerr << "[ERROR] Entries for createMultiTemplates should only contain a single image." << endl;
            raise(SIGTERM);
        }
        Image image;
        string imagePath = tokens[1];
        string desc = tokens[2];
        if (!readImage(imagePath, image)) {
            cerr << "[ERROR] Failed to load image file: " << imagePath << "." << endl;
            raise(SIGTERM);
        }
        image.description = mapStringToImgLabel[desc];

        vector<vector<uint8_t>> templs;
        vector<EyePair> eyes;
        auto ret = implPtr->createTemplate(image, role, templs, eyes);

        /* Check that function is implemented */
        if (ret.code == ReturnCode::NotImplemented) {
            cerr << "[ERROR] The createTemplate(image, role, templs, eyes) function returned ReturnCode::NotImplemented.  This function must be implemented!" << std::endl;
            raise(SIGTERM);
        }

        if (templs.size() == 0) {
            cerr << "[ERROR] The output template vector must contain at least one template." << endl;
            raise(SIGTERM);
        }
        if (templs.size() != eyes.size()) {
            cerr << "[ERROR] The number of eye coordinates do not match the number of templates." << endl;
            raise(SIGTERM);
        }

        for (unsigned int i = 0; i < templs.size(); i++) {
            /* Open template file for writing */
            string templFile{id + "_" + to_string(i) + ".template"};
            ofstream templStream(templatesDir + "/" + templFile);
            if (!templStream.is_open()) {
                cerr << "[ERROR] Failed to open stream for " << templatesDir + "/" + templFile << "." << endl;
                raise(SIGTERM);
            }
            /* Write template file */
            auto templ = templs[i];
            templStream.write((char*)templ.data(), templ.size());

            /* Write template stats to log */
            logStream << id << "_" << i << " "
                << imagePath << " "
                << templ.size() << " "
                << static_cast<std::underlying_type<ReturnCode>::type>(ret.code) << " "
                << templs.size() << " "
                << i << " "
                << (eyes.size() > 0 ? eyes[i].isLeftAssigned : false) << " "
                << (eyes.size() > 0 ? eyes[i].isRightAssigned : false) << " "
                << (eyes.size() > 0 ? eyes[i].xleft : 0) << " "
                << (eyes.size() > 0 ? eyes[i].yleft : 0) << " "
                << (eyes.size() > 0 ? eyes[i].xright : 0) << " "
                << (eyes.size() > 0 ? eyes[i].yright : 0)
                << endl;
        }
    }
    inputStream.close();

    /* Remove the input file */
    if( remove(inputFile.c_str()) != 0 )
        cerr << "Error deleting file: " << inputFile << endl;

    return SUCCESS;
}

int
match(
        std::shared_ptr<Interface> &implPtr,
        const string &inputFile,
        const string &templatesDir,
        const string &scoresLog)
{
    /* Read probes */
    ifstream inputStream(inputFile);
    if (!inputStream.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << inputFile << "." << endl;
        raise(SIGTERM);
    }

    /* Open scores log for writing */
    ofstream scoresStream(scoresLog);
    if (!scoresStream.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << scoresLog << "." << endl;
        raise(SIGTERM);
    }
    /* header */
    scoresStream << "enrollTempl verifTempl simScore returnCode" << endl;

    /* Process each probe */
    string enrollID, verifID;
    while (inputStream >> enrollID >> verifID) {
        vector<uint8_t> enrollTempl, verifTempl;
        double similarity = -1.0;
        /* Read templates from file */
        if (readTemplateFromFile(templatesDir + "/" + enrollID, enrollTempl) != SUCCESS) {
            cerr << "[ERROR] Unable to retrieve template from file : "
                    << templatesDir + "/" + enrollID << endl;
            raise(SIGTERM);
        }
        if (readTemplateFromFile(templatesDir + "/" + verifID, verifTempl) != SUCCESS) {
            cerr << "[ERROR] Unable to retrieve template from file : "
                    << templatesDir + "/" + verifID << endl;
            raise(SIGTERM);
        }

        /* Call match */
        auto ret = implPtr->matchTemplates(verifTempl, enrollTempl, similarity);

        /* Write to scores log file */
        scoresStream << enrollID << " "
                << verifID << " "
                << similarity << " "
                << static_cast<std::underlying_type<ReturnCode>::type>(ret.code)
                << endl;
    }
    inputStream.close();

    /* Remove the input file */
    if( remove(inputFile.c_str()) != 0 )
        cerr << "Error deleting file: " << inputFile << endl;

    return SUCCESS;
}

void usage(const string &executable)
{
    cerr << "Usage: " << executable << " createTemplate -x enroll|verif -c configDir "
            "-o outputDir -h outputStem -i inputFile -t numForks -j templatesDir" << endl;
    cerr << "       " << executable << " match -c configDir "
                "-o outputDir -h outputStem -i inputFile -t numForks -j templatesDir" << endl;
    exit(EXIT_FAILURE);
}

int
main(
        int argc,
        char* argv[])
{
    auto exitStatus = SUCCESS;

    uint16_t currAPIMajorVersion{5},
		currAPIMinorVersion{0},
		currStructsMajorVersion{1},
		currStructsMinorVersion{2};

    /* Check versioning of both frvt_structs.h and API header file */
	if ((FRVT::FRVT_STRUCTS_MAJOR_VERSION != currStructsMajorVersion) ||
			(FRVT::FRVT_STRUCTS_MINOR_VERSION != currStructsMinorVersion)) {
		cerr << "[ERROR] You've compiled your library with an old version of the frvt_structs.h file: version " <<
		    FRVT::FRVT_STRUCTS_MAJOR_VERSION << "." <<
		    FRVT::FRVT_STRUCTS_MINOR_VERSION <<
		    ".  Please re-build with the latest version: " <<
		    currStructsMajorVersion << "." <<
	   	    currStructsMinorVersion << "." << endl;
		return (FAILURE);
	}

	if ((FRVT_11::API_MAJOR_VERSION != currAPIMajorVersion) ||
			(FRVT_11::API_MINOR_VERSION != currAPIMinorVersion)) {
		std::cerr << "[ERROR] You've compiled your library with an old version of the API header file: " <<
		    FRVT_11::API_MAJOR_VERSION << "." <<
		    FRVT_11::API_MINOR_VERSION <<
		    ".  Please re-build with the latest version:" <<
		    currAPIMajorVersion << "." <<
		    currStructsMinorVersion << "." << endl;
		return (FAILURE);
	}

    int requiredArgs = 2; /* exec name and action */
    if (argc < requiredArgs)
        usage(argv[0]);

    string actionstr{argv[1]},
        configDir{"config"},
        outputDir{"output"},
        outputFileStem{"stem"},
        inputFile,
        templatesDir,
	roleStr{""};
    int numForks = 1;

    for (int i = 0; i < argc - requiredArgs; i++) {
        if (strcmp(argv[requiredArgs+i],"-c") == 0)
            configDir = argv[requiredArgs+(++i)];
        else if (strcmp(argv[requiredArgs+i],"-o") == 0)
            outputDir = argv[requiredArgs+(++i)];
        else if (strcmp(argv[requiredArgs+i],"-h") == 0)
            outputFileStem = argv[requiredArgs+(++i)];
        else if (strcmp(argv[requiredArgs+i],"-i") == 0)
            inputFile = argv[requiredArgs+(++i)];
        else if (strcmp(argv[requiredArgs+i],"-j") == 0)
            templatesDir = argv[requiredArgs+(++i)];
        else if (strcmp(argv[requiredArgs+i],"-t") == 0)
            numForks = atoi(argv[requiredArgs+(++i)]);
        else if (strcmp(argv[requiredArgs+i],"-x") == 0)
	    roleStr = argv[requiredArgs+(++i)];
        else {
            cerr << "[ERROR] Unrecognized flag: " << argv[requiredArgs+i] << endl;;
            usage(argv[0]);
        }
    }

    Action action = mapStringToAction[actionstr];
    switch (action) {
        case Action::CreateTemplate:
        case Action::CreateMultiTemplates:
        case Action::Match:
            break;
        default:
            cerr << "[ERROR] Unknown command: " << actionstr << endl;
            usage(argv[0]);
    }

    TemplateRole role{};
    if (action == Action::CreateTemplate || action == Action::CreateMultiTemplates) {
    	if (roleStr == "enroll")
    		role = TemplateRole::Enrollment_11;
    	else if (roleStr == "verif")
    		role = TemplateRole::Verification_11;
    	else {
            cerr << "Unknown template role: " << roleStr << endl;
            usage(argv[0]);
    	}
    }

    /* Get implementation pointer */
    auto implPtr = Interface::getImplementation();
    /* Initialization */
    auto ret = implPtr->initialize(configDir);
    if (ret.code != ReturnCode::Success) {
        cerr << "[ERROR] initialize() returned error: "
                << ret.code << "." << endl;
        return FAILURE;
    }

    /* Split input file into appropriate number of splits */
    vector<string> inputFileVector;
    if (splitInputFile(inputFile, outputDir, numForks, inputFileVector) != SUCCESS) {
        cerr << "[ERROR] An error occurred with processing the input file." << endl;
        return FAILURE;
    }
    
    bool parent = false;
	int i = 0;
    for (auto &inputFile : inputFileVector) {
		/* Fork */
		switch(fork()) {
		case 0: /* Child */
			if (action == Action::CreateTemplate)
				return createTemplate(
						implPtr,
						inputFile,
						outputDir + "/" + outputFileStem + ".log." + to_string(i),
						templatesDir,
						role);
            else if (action == Action::CreateMultiTemplates)
                return createMultiTemplates(
                        implPtr,
                        inputFile,
                        outputDir + "/" + outputFileStem + ".log." + to_string(i),
                        templatesDir,
                        role);
			else if (action == Action::Match)
				return match(
						implPtr,
						inputFile,
						templatesDir,
						outputDir + "/" + outputFileStem + ".log." + to_string(i));
		case -1: /* Error */
			cerr << "Problem forking" << endl;
			break;
		default: /* Parent */
			parent = true;
			break;
		}
		i++;
	}


    /* Parent -- wait for children */
    if (parent) {
        while (numForks > 0) {
            int stat_val;
            pid_t cpid;

            cpid = wait(&stat_val);
            if (WIFEXITED(stat_val)) { exitStatus = WEXITSTATUS(stat_val); }
            else if (WIFSIGNALED(stat_val)) {
                cerr << "PID " << cpid << " exited due to signal " <<
                        WTERMSIG(stat_val) << endl;
                exitStatus = FAILURE;
            } else {
                cerr << "PID " << cpid << " exited with unknown status." << endl;
                exitStatus = FAILURE;
            }
            numForks--;
        }
    }

    return exitStatus;
}
