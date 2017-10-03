/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package binaryclassifierui;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Application;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.Stage;

/**
 *
 * @author Chris
 */
public class BinaryClassifierUI extends Application {
    
    @Override
    public void start(Stage primaryStage) {
        DataProcessing dp = new DataProcessing();
        Text prompt = new Text("Start by choosing your options and pressing");
        Text prompt2 = new Text("the button of your choice (perceptron/naive");
        Text command1 = new Text("Accuracy:");
        Text command2 = new Text("");
        Text command3 = new Text("Precision:");
        Text command4 = new Text("");
        Text command5 = new Text("Recall:");
        Text command6 = new Text("");
        Text command7 = new Text("");
        ComboBox cb1 = new ComboBox();
        ComboBox cb2 = new ComboBox();
        cb1.getItems().addAll("Frequency");
        cb2.getItems().addAll("No Punctuation", "TF-IDF", "Default");
        Button b1 = new Button ("Perceptron");
        b1.setOnAction(e -> {
            command7.setText("Running");
            if (cb1.getValue().equals("Frequency")) {
                if (cb2.getValue().equals("Default")) {
                    try {
                        double[] doubleArray = dp.trainPerceptron();
                        command2.setText(Double.toString(doubleArray[0]));
                        command4.setText(Double.toString(doubleArray[1]));
                        command6.setText(Double.toString(doubleArray[2]));
                    } catch (IOException ex) {
                        System.out.println("same");
                    }
                }
                if (cb2.getValue().equals("TF-IDF")) {
                    try {
                        double[] doubleArray = dp.trainPerceptronTF();
                        command2.setText(Double.toString(doubleArray[0]));
                        command4.setText(Double.toString(doubleArray[1]));
                        command6.setText(Double.toString(doubleArray[2]));
                    } catch (IOException ex) {}
                }
                if (cb2.getValue().equals("No Punctuation")) {
                    try {
                        double[] doubleArray = dp.trainPerceptronNoPunc();
                        command2.setText(Double.toString(doubleArray[0]));
                        command4.setText(Double.toString(doubleArray[1]));
                        command6.setText(Double.toString(doubleArray[2]));
                    } catch (IOException ex) {}
                }
            }
            command7.setText("Finished");
        });
        Button b2 = new Button ("Naive Bayes");
        b2.setOnAction(e -> {
            command7.setText("Running");
            if (cb1.getValue().equals("Frequency")) {
                if (cb2.getValue().equals("Default")) {
                    try {
                        double[] doubleArray = dp.trainNaiveBayes();
                        command2.setText(Double.toString(dp.trainNaiveBayes()[0]));
                        command4.setText(Double.toString(doubleArray[1]));
                        command6.setText(Double.toString(doubleArray[2]));
                    } catch (IOException ex) {
                        System.out.println("same");
                    }
                }
                if (cb2.getValue().equals("TF-IDF")) {
                    try {
                        double[] doubleArray = dp.trainNaiveBayesTF();
                        command2.setText(Double.toString(doubleArray[0]));
                        command4.setText(Double.toString(doubleArray[1]));
                        command6.setText(Double.toString(doubleArray[2]));
                    } catch (IOException ex) {}
                }
                if (cb2.getValue().equals("No Punctuation")) {
                    try {
                        double[] doubleArray = dp.trainNaiveBayesNoPunc();
                        command2.setText(Double.toString(doubleArray[0]));
                        command4.setText(Double.toString(doubleArray[1]));
                        command6.setText(Double.toString(doubleArray[2]));
                    } catch (IOException ex) {}
                }
            }
            command7.setText("Finished");
        });
        StackPane root = new StackPane();
        HBox hbox = new HBox();
        HBox hbox1 = new HBox();
        VBox vbox = new VBox();
        hbox1.getChildren().addAll(b1, b2);
        hbox.getChildren().addAll(cb1, cb2);
        vbox.getChildren().addAll(prompt, prompt2, hbox, hbox1, command1, command2, command3, command4, command5, command6, command7);
        root.getChildren().addAll(vbox);
        
        Scene scene = new Scene(root, 300, 250);
        
        primaryStage.setTitle("Classifier");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        launch(args);
    }
    
}
