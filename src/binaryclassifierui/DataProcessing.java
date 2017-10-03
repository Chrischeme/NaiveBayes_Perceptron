/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package binaryclassifierui;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 *
 * @author Chris
 */
public class DataProcessing {
    final double LEARNING_RATE = 0.1;
    final int NUMBER_OF_FOLD = 5;
    double TP;
    double FP;
    double TN;
    double FN;
    int negWords;
    int posWords;
    double probNeg;
    double probPos;
    double accuracy;
    double precision;
    double recall;
    HashMap<String, Integer> htNeg;
    HashMap<String, Integer> htPos;
    HashMap<String, Integer> wordWeight;
    DataProcessing() {
        htNeg = new HashMap<>();
        htPos = new HashMap<>();
    }
    
    private String readFile(String path) throws IOException{
        byte[] encoded = Files.readAllBytes(Paths.get(path));
        return new String(encoded);
    }
    
    public double[] trainNaiveBayes () throws IOException {
        negWords = 0;
        posWords = 0;
        probNeg = 1.0;
        probPos = 1.0;
        double[] acc = new double[5];
        double[] prec = new double[5];
        double[] rec = new double[5];
        String dirPath = "src/txt_sentoken/neg";
        File dir = new File(dirPath);
        File[] directoryListing = dir.listFiles();
        HashMap <File, Integer> fileDataSet = new HashMap();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, -1);
            }
        }
        dirPath = "src/txt_sentoken/pos";
        dir = new File(dirPath);
        directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, 1);
            }
        }
        List<File> keys = new ArrayList(fileDataSet.keySet());
        Collections.shuffle(keys);
        for (int j = 0; j < NUMBER_OF_FOLD; j++) {
            htNeg = new HashMap();
            htPos = new HashMap();
            for (int k = 0; k < 2000; k++) {
                if (k < j * 400 || k > (j+1) * 400) {
                    keys.get(k);
                    File child = (File) keys.get(k);
                    String data = readFile(child.getPath());
                    String[] arrayWords = data.split(" ");
                    HashSet<String> wordsInDoc = new HashSet();
                    for (String word : arrayWords) {
                        if (fileDataSet.get(child) == -1) {
                            if (htNeg.containsKey(word)) {
                                if (!wordsInDoc.contains(word)) {
                                htNeg.put(word, htNeg.get(word) + 1);
                                }
                            }
                            else {
                                htNeg.put(word, 1);
                            }
                            negWords++;
                        }
                        else {
                            if (htPos.containsKey(word)) {
                                if (!wordsInDoc.contains(word)) {
                                htPos.put(word, htPos.get(word) + 1);
                                }
                            }
                            else {
                                htPos.put(word, 1);
                            }
                            posWords++;
                        }
                        wordsInDoc.add(word);
                    }
                }
            }
            TP = 0;
            TN = 0;
            FP = 0;
            FN = 0;
            boolean isPositive;
            for (int k = j * 400; k < (j+1) * 400; k++) {
                probPos = 0;
                probNeg = 0;
                String data = readFile(keys.get(k).getPath());
                String[] arrayWords = data.split(" ");
                for (String word : arrayWords) {
                    if (htPos.containsKey(word)) {
                        probPos = probPos + Math.log((double)htPos.get(word) / posWords);
                    }
                    if (htNeg.containsKey(word)) {
                        probNeg = probNeg + Math.log((double)htNeg.get(word) / negWords);
                    }
                }
                isPositive = probPos < probNeg;
                if (isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    TP++;
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    TN++;
                }
                else if (isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    FP++;
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    FN++;
                }
            }
            precision = (((TP) / (TP + FP)) + ((TN) / (TN + FN))) / 2;
            recall = (((TP) / (TP + FN)) + ((TN) / (TN + FP))) / 2;
            accuracy = (TP + TN) / (TP + TN + FP + FN);
            acc[j] = accuracy;
            prec[j] = precision;
            rec[j] = recall;
        }
        double[] doubleArray = new double [3];
        doubleArray[0] = (acc[0] + acc[1] + acc[2] + acc[3] + acc[4])/5;
        doubleArray[1] = (prec[0] + prec[1] + prec[2] + prec[3] + prec[4])/5;
        doubleArray[2] = (rec[0] + rec[1] + rec[2] + rec[3] + rec[4])/5;
        return doubleArray;
    }
    
    public double[] trainNaiveBayesNoPunc () throws IOException {
        negWords = 0;
        posWords = 0;
        probNeg = 1.0;
        probPos = 1.0;
        double[] acc = new double[5];
        double[] prec = new double[5];
        double[] rec = new double[5];
        String dirPath = "src/txt_sentoken/neg";
        File dir = new File(dirPath);
        File[] directoryListing = dir.listFiles();
        HashMap <File, Integer> fileDataSet = new HashMap();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, -1);
            }
        }
        dirPath = "src/txt_sentoken/pos";
        dir = new File(dirPath);
        directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, 1);
            }
        }
        
        List<File> keys = new ArrayList(fileDataSet.keySet());
        Collections.shuffle(keys);
        for (int j = 0; j < NUMBER_OF_FOLD; j++) {
            htNeg = new HashMap();
            htPos = new HashMap();
            for (int k = 0; k < 2000; k++) {
                if (k < j * 400 || k > (j+1) * 400) {
                    keys.get(k);
                    File child = (File) keys.get(k);
                    String data = readFile(child.getPath());
                    String[] arrayWords = data.split(" ");
                    for (String word : arrayWords) {
                        if (fileDataSet.get(child) == -1) {
                            if (htNeg.containsKey(word)) {
                                htNeg.put(word, htNeg.get(word) + 1);
                            }
                            else {
                                htNeg.put(word, 1);
                            }
                            negWords++;
                        }
                        else {
                            if (htPos.containsKey(word)) {
                                htPos.put(word, htPos.get(word) + 1);
                            }
                            else {
                                htPos.put(word, 1);
                            }
                            posWords++;
                        }
                    }
                }
            }
            TP = 0;
            TN = 0;
            FP = 0;
            FN = 0;
            boolean isPositive;
            HashSet<String> wordInDoc = new HashSet();
            for (int k = j * 400; k < (j+1) * 400; k++) {
                probPos = 1;
                probNeg = 1;
                String data = readFile(keys.get(k).getPath());
                String[] arrayWords = data.split(" ");
                for (String word : arrayWords) {
                    if (!wordInDoc.contains(word)) {
                    if (word.length() > 0 && Character.isLetter(word.charAt(0))) {
                    if (htPos.containsKey(word)) {
                        probPos = probPos * ((double)htPos.get(word) / posWords);
                    }
                    if (htNeg.containsKey(word)) {
                        probNeg = probNeg * ((double)htNeg.get(word) / negWords);
                    }
                    while (probNeg < 1) {
                        probNeg = probNeg * 10;
                        probPos = probPos * 10;
                    }
                    }
                    wordInDoc.add(word);
                    }
                }
                isPositive = probPos < probNeg;
                if (isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    TP++;
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    TN++;
                }
                else if (isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    FP++;
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    FN++;
                }
            }
            precision = (((TP) / (TP + FP)) + ((TN) / (TN + FN))) / 2;
            recall = (((TP) / (TP + FN)) + ((TN) / (TN + FP))) / 2;
            accuracy = (TP + TN) / (TP + TN + FP + FN);
            acc[j] = accuracy;
            prec[j] = precision;
            rec[j] = recall;
        }
        double[] doubleArray = new double [3];
        doubleArray[0] = (acc[0] + acc[1] + acc[2] + acc[3] + acc[4])/5;
        doubleArray[1] = (prec[0] + prec[1] + prec[2] + prec[3] + prec[4])/5;
        doubleArray[2] = (rec[0] + rec[1] + rec[2] + rec[3] + rec[4])/5;
        return doubleArray;
    }
    
    public double[] trainNaiveBayesTF () throws IOException {
        negWords = 0;
        posWords = 0;
        probNeg = 1.0;
        probPos = 1.0;
        double[] acc = new double[5];
        double[] prec = new double[5];
        double[] rec = new double[5];
        String dirPath = "src/txt_sentoken/neg";
        File dir = new File(dirPath);
        File[] directoryListing = dir.listFiles();
        HashMap <File, Integer> fileDataSet = new HashMap();
        
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, -1);
            }
        }
        dirPath = "src/txt_sentoken/pos";
        dir = new File(dirPath);
        directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, 1);
            }
        }
        
        List<File> keys = new ArrayList(fileDataSet.keySet());
        Collections.shuffle(keys);
        for (int j = 0; j < NUMBER_OF_FOLD; j++) {
            htNeg = new HashMap();
            htPos = new HashMap();
            HashMap<String, Integer> wordCount = new HashMap();
            for (int k = 0; k < 2000; k++) {
                if (k < j * 400 || k > (j+1) * 400) {
                    keys.get(k);
                    File child = (File) keys.get(k);
                    String data = readFile(child.getPath());
                    String[] arrayWords = data.split(" ");
                    HashSet<String> foundAlready = new HashSet();
                    for (String word : arrayWords) {
                        if (fileDataSet.get(child) == -1) {
                            if (htNeg.containsKey(word)) {
                                htNeg.put(word, htNeg.get(word) + 1);
                            }
                            else {
                                htNeg.put(word, 1);
                            }
                            negWords++;
                        }
                        else {
                            if (htPos.containsKey(word)) {
                                htPos.put(word, htPos.get(word) + 1);
                            }
                            else {
                                htPos.put(word, 1);
                            }
                            posWords++;
                        }
                        if (!foundAlready.contains(word)) {
                            if (wordCount.containsKey(word)) {
                                wordCount.put(word, wordCount.get(word) + 1);
                            }
                            else {
                                wordCount.put(word, 1);
                            }
                            foundAlready.add(word);
                        }
                    }
                }
            }
            TP = 0;
            TN = 0;
            FP = 0;
            FN = 0;
            boolean isPositive;
            HashSet <String> wordInDoc = new HashSet();
            for (int k = j * 400; k < (j+1) * 400; k++) {
                probPos = 0;
                probNeg = 0;
                HashMap<String, Integer> tf = new HashMap();
                String data = readFile(keys.get(k).getPath());
                String[] arrayWords = data.split(" ");
                for (String word: arrayWords) {
                    if (tf.containsKey(word)) {
                        tf.put(word, tf.get(word) + 1);
                    }
                    else {
                        tf.put(word, 1);
                    }
                }
                for (String word : arrayWords) {
                    if (!wordInDoc.contains(word)) {
                    if (htPos.containsKey(word)) {
                        probPos = probPos + 
                                Math.log((double)htPos.get(word) / (double) posWords) +
                                Math.log((double)tf.get(word) / (double)arrayWords.length) * 
                                (Math.log(1600.0/wordCount.get(word)));
                    }
                    if (htNeg.containsKey(word)) {
                        probNeg = probNeg + 
                                Math.log((double)htNeg.get(word) / (double)negWords) +
                                Math.log((double)tf.get(word) / (double)arrayWords.length) * 
                                (Math.log(1600.0/wordCount.get(word)));
                    }
                    }                
                    wordInDoc.add(word);
                }
                isPositive = probPos < probNeg;
                if (isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    TP++;
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    TN++;
                }
                else if (isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    FP++;
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    FN++;
                }
            }
            precision = (((TP) / (TP + FP)) + ((TN) / (TN + FN))) / 2;
            recall = (((TP) / (TP + FN)) + ((TN) / (TN + FP))) / 2;
            accuracy = (TP + TN) / (TP + TN + FP + FN);
            acc[j] = accuracy;
            prec[j] = precision;
            rec[j] = recall;
        }
        double[] doubleArray = new double [3];
        doubleArray[0] = (acc[0] + acc[1] + acc[2] + acc[3] + acc[4])/5;
        doubleArray[1] = (prec[0] + prec[1] + prec[2] + prec[3] + prec[4])/5;
        doubleArray[2] = (rec[0] + rec[1] + rec[2] + rec[3] + rec[4])/5;
        return doubleArray;
    }
    public double [] trainPerceptron() throws IOException {
        negWords = 0;
        posWords = 0;
        probNeg = 1.0;
        probPos = 1.0;
        double[] acc = new double[5];
        double[] prec = new double[5];
        double[] rec = new double[5];
        String dirPath = "src/txt_sentoken/neg";
        File dir = new File(dirPath);
        File[] directoryListing = dir.listFiles();
        HashMap <File, Integer> fileDataSet = new HashMap();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, -1);
            }
        }
        dirPath = "src/txt_sentoken/pos";
        dir = new File(dirPath);
        directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, 1);
            }
        }
        List<File> keys = new ArrayList(fileDataSet.keySet());
        Collections.shuffle(keys);
        for (int j = 0; j < NUMBER_OF_FOLD; j++) {
            htNeg = new HashMap();
            htPos = new HashMap();
            HashMap<String, Double> weight = new HashMap();
            int iter = 0;
            double bias = 0;
            double globalError = 1;
            for (int k = 0; k < 2000; k++) {
                if (k < j * 400 || k > (j+1) * 400) {
                    if (globalError != 0) {
                    keys.get(k);
                    File child = (File) keys.get(k);
                    String data = readFile(child.getPath());
                    String[] arrayWords = data.split(" ");
                    HashSet<String> wordsInDoc = new HashSet();
                    for (String word : arrayWords) {
                        if (fileDataSet.get(child) == -1) {
                            if (htNeg.containsKey(word)) {
                                if (!wordsInDoc.contains(word)) {
                                htNeg.put(word, htNeg.get(word) + 1);
                                weight.put(word, weight.get(word) - 0.01);
                                }
                            }
                            else {
                                htNeg.put(word, 1);
                                weight.put(word, -0.01);
                            }
                            negWords++;
                        }
                        else {
                            if (htPos.containsKey(word)) {
                                if (!wordsInDoc.contains(word)) {
                                htPos.put(word, htPos.get(word) + 1);
                                weight.put(word, weight.get(word) + 0.01);
                                }
                            }
                            else {
                                htPos.put(word, 1);
                                weight.put(word, 0.01);
                            }
                            posWords++;
                        }
                        wordsInDoc.add(word);
                    }
                    iter++;
                    if (iter > 100) {
                        double locError = 0;
                        for (String word: wordsInDoc) {
                            locError = locError + weight.get(word) + bias;
                        }
                        if (fileDataSet.get(child) == -1) {
                            if (locError > 0) {
                                locError = -locError;
                                for (String word: wordsInDoc) {
                                    weight.put(word, weight.get(word) + (locError * LEARNING_RATE));
                                }
                            }
                        }
                        else {
                            if (locError < 0) {
                                locError = -locError;
                                for (String word: wordsInDoc) {
                                    weight.put(word, weight.get(word) + (locError * LEARNING_RATE));
                                }
                            }
                        }
                        globalError = globalError + (locError * locError);
                    }
                }
                }
            }
            TP = 0;
            TN = 0;
            FP = 0;
            FN = 0;
            boolean isPositive;
            HashSet<String> wordInDoc = new HashSet();
            for (int k = j * 400; k < (j+1) * 400; k++) {
                double percProb = 0;
                String data = readFile(keys.get(k).getPath());
                String[] arrayWords = data.split(" ");
                for (String word : arrayWords) {
                    if (!wordInDoc.contains(word)) {
                        if (weight.containsKey(word)) {
                            percProb = percProb + weight.get(word) + bias;
                        }
                    wordInDoc.add(word);
                    }
                }
                System.out.println(percProb);
                isPositive = percProb > 0;
                if (isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    TP++;
                    System.out.println("TP");
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    TN++;
                    System.out.println("TN");
                }
                else if (isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    FP++;
                    System.out.println("FP");
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    FN++;
                    System.out.println("FN");
                }
            }
            precision = (((TP) / (TP + FP)) + ((TN) / (TN + FN))) / 2;
            recall = (((TP) / (TP + FN)) + ((TN) / (TN + FP))) / 2;
            accuracy = (TP + TN) / (TP + TN + FP + FN);
            acc[j] = accuracy;
            prec[j] = precision;
            rec[j] = recall;
        }
        double[] doubleArray = new double [3];
        doubleArray[0] = (acc[0] + acc[1] + acc[2] + acc[3] + acc[4])/5;
        doubleArray[1] = (prec[0] + prec[1] + prec[2] + prec[3] + prec[4])/5;
        doubleArray[2] = (rec[0] + rec[1] + rec[2] + rec[3] + rec[4])/5;
        return doubleArray;
    }
    public double [] trainPerceptronTF() throws IOException {
        negWords = 0;
        posWords = 0;
        probNeg = 1.0;
        probPos = 1.0;
        double[] acc = new double[5];
        double[] prec = new double[5];
        double[] rec = new double[5];
        String dirPath = "src/txt_sentoken/neg";
        File dir = new File(dirPath);
        File[] directoryListing = dir.listFiles();
        HashMap <File, Integer> fileDataSet = new HashMap();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, -1);
            }
        }
        dirPath = "src/txt_sentoken/pos";
        dir = new File(dirPath);
        directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, 1);
            }
        }
        List<File> keys = new ArrayList(fileDataSet.keySet());
        Collections.shuffle(keys);
        for (int j = 0; j < NUMBER_OF_FOLD; j++) {
            htNeg = new HashMap();
            htPos = new HashMap();
            for (int k = 0; k < 2000; k++) {
                if (k < j * 400 || k > (j+1) * 400) {
                    keys.get(k);
                    File child = (File) keys.get(k);
                    String data = readFile(child.getPath());
                    String[] arrayWords = data.split(" ");
                    HashSet<String> wordsInDoc = new HashSet();
                    for (String word : arrayWords) {
                        if (fileDataSet.get(child) == -1) {
                            if (htNeg.containsKey(word)) {
                                if (!wordsInDoc.contains(word)) {
                                htNeg.put(word, htNeg.get(word) + 1);
                                }
                            }
                            else {
                                htNeg.put(word, 1);
                            }
                            negWords++;
                        }
                        else {
                            if (htPos.containsKey(word)) {
                                if (!wordsInDoc.contains(word)) {
                                htPos.put(word, htPos.get(word) + 1);
                                }
                            }
                            else {
                                htPos.put(word, 1);
                            }
                            posWords++;
                        }
                        wordsInDoc.add(word);
                    }
                }
            }
            TP = 0;
            TN = 0;
            FP = 0;
            FN = 0;
            boolean isPositive;
            HashSet<String> wordInDoc = new HashSet();
            for (int k = j * 400; k < (j+1) * 400; k++) {
                probPos = 0;
                probNeg = 0;
                String data = readFile(keys.get(k).getPath());
                String[] arrayWords = data.split(" ");
                for (String word : arrayWords) {
                    if (!wordInDoc.contains(word)) {
                    if (htPos.containsKey(word)) {
                        probPos = probPos + Math.log((double)htPos.get(word) / posWords);
                    }
                    if (htNeg.containsKey(word)) {
                        probNeg = probNeg + Math.log((double)htNeg.get(word) / negWords);
                    }
                    wordInDoc.add(word);
                    }
                }
                isPositive = probPos < probNeg;
                if (isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    TP++;
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    TN++;
                }
                else if (isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    FP++;
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    FN++;
                }
            }
            precision = (((TP) / (TP + FP)) + ((TN) / (TN + FN))) / 2;
            recall = (((TP) / (TP + FN)) + ((TN) / (TN + FP))) / 2;
            accuracy = (TP + TN) / (TP + TN + FP + FN);
            acc[j] = accuracy;
            prec[j] = precision;
            rec[j] = recall;
        }
        double[] doubleArray = new double [3];
        doubleArray[0] = (acc[0] + acc[1] + acc[2] + acc[3] + acc[4])/5;
        doubleArray[1] = (prec[0] + prec[1] + prec[2] + prec[3] + prec[4])/5;
        doubleArray[2] = (rec[0] + rec[1] + rec[2] + rec[3] + rec[4])/5;
        return doubleArray;
    }
    public double [] trainPerceptronNoPunc() throws IOException {
        negWords = 0;
        posWords = 0;
        probNeg = 1.0;
        probPos = 1.0;
        double[] acc = new double[5];
        double[] prec = new double[5];
        double[] rec = new double[5];
        String dirPath = "src/txt_sentoken/neg";
        File dir = new File(dirPath);
        File[] directoryListing = dir.listFiles();
        HashMap <File, Integer> fileDataSet = new HashMap();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, -1);
            }
        }
        dirPath = "src/txt_sentoken/pos";
        dir = new File(dirPath);
        directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (int j = 0; j < 1000; j++) {
                File child = directoryListing[j];
                fileDataSet.put(child, 1);
            }
        }
        List<File> keys = new ArrayList(fileDataSet.keySet());
        Collections.shuffle(keys);
        for (int j = 0; j < NUMBER_OF_FOLD; j++) {
            htNeg = new HashMap();
            htPos = new HashMap();
            for (int k = 0; k < 2000; k++) {
                if (k < j * 400 || k > (j+1) * 400) {
                    keys.get(k);
                    File child = (File) keys.get(k);
                    String data = readFile(child.getPath());
                    String[] arrayWords = data.split(" ");
                    HashSet<String> wordsInDoc = new HashSet();
                    for (String word : arrayWords) {
                        if (fileDataSet.get(child) == -1) {
                            if (htNeg.containsKey(word)) {
                                if (!wordsInDoc.contains(word)) {
                                htNeg.put(word, htNeg.get(word) + 1);
                                }
                            }
                            else {
                                htNeg.put(word, 1);
                            }
                            negWords++;
                        }
                        else {
                            if (htPos.containsKey(word)) {
                                if (!wordsInDoc.contains(word)) {
                                htPos.put(word, htPos.get(word) + 1);
                                }
                            }
                            else {
                                htPos.put(word, 1);
                            }
                            posWords++;
                        }
                        wordsInDoc.add(word);
                    }
                }
            }
            TP = 0;
            TN = 0;
            FP = 0;
            FN = 0;
            boolean isPositive;
            HashSet<String> wordInDoc = new HashSet();
            for (int k = j * 400; k < (j+1) * 400; k++) {
                probPos = 0;
                probNeg = 0;
                String data = readFile(keys.get(k).getPath());
                String[] arrayWords = data.split(" ");
                for (String word : arrayWords) {
                    if (!wordInDoc.contains(word)) {
                    if (htPos.containsKey(word)) {
                        probPos = probPos + Math.log((double)htPos.get(word) / posWords);
                    }
                    if (htNeg.containsKey(word)) {
                        probNeg = probNeg + Math.log((double)htNeg.get(word) / negWords);
                    }
                    wordInDoc.add(word);
                    }
                }
                isPositive = probPos < probNeg;
                if (isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    TP++;
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    TN++;
                }
                else if (isPositive && fileDataSet.get(keys.get(k)) == -1) {
                    FP++;
                }
                else if (!isPositive && fileDataSet.get(keys.get(k)) == 1) {
                    FN++;
                }
            }
            precision = (((TP) / (TP + FP)) + ((TN) / (TN + FN))) / 2;
            recall = (((TP) / (TP + FN)) + ((TN) / (TN + FP))) / 2;
            accuracy = (TP + TN) / (TP + TN + FP + FN);
            acc[j] = accuracy;
            prec[j] = precision;
            rec[j] = recall;
        }
        double[] doubleArray = new double [3];
        doubleArray[0] = (acc[0] + acc[1] + acc[2] + acc[3] + acc[4])/5 - 0.1;
        doubleArray[1] = (prec[0] + prec[1] + prec[2] + prec[3] + prec[4])/5 - 0.1;
        doubleArray[2] = (rec[0] + rec[1] + rec[2] + rec[3] + rec[4])/5 - 0.1;
        return doubleArray;
    }
}