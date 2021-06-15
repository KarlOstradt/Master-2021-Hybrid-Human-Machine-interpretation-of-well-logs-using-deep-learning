package csv;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Class for manipulating, loading and saving datasets on the CSV format.
 *
 * @author Karl Ostradt
 */
public final class CSV
{
  /** A list of columns. */
  private List<List<Object>> columns_;

  /** The names of the columns. */
  private List<String> names_;

  /** The delimiter to separate values. */
  private String delimiter_;

  /**
   * Constructor for creating an empty dataset.
   */
  public CSV()
  {
    delimiter_ = ",";
    names_ = new ArrayList<String>();
    columns_ = new ArrayList<List<Object>>();
  }

  /**
   * Constructor for creating a dataset with given columns.
   * @param columnNames The names of each column in correct order. Non-null.
   * @param columns  A list of columns. Each column is a list of Double values. Non-null.
   */
  public CSV(List<String> columnNames, List<List<Object>> columns)
  {
    if (columnNames == null)
      throw new IllegalArgumentException("columnNames cannot be null");
    if (columns == null)
      throw new IllegalArgumentException("columns cannot be null");
    if (columnNames.size() != columns.size())
      throw new IllegalArgumentException("columnNames and columns must be of same size");

    delimiter_ = ",";
    names_ = columnNames;
    columns_ = columns;
  }

  /**
   * Adds a column to the dataset.
   * @param name The name of the column. Cannot be empty or null.
   * @param column A list of values. Non-null.
   */
  public void addColumn(String name, List<Object> column)
  {
    if (name == null)
      throw new IllegalArgumentException("name cannot be null");
    if (name.isEmpty())
      throw new IllegalArgumentException("name cannot be empty");
    if (column == null)
      throw new IllegalArgumentException("column cannot be null");
    columns_.add(column);
    names_.add(name);
  }

  /**
   * Load a CSV dataset from file. Uses default delimiter ',' (comma).
   *
   * @param file The file to read from. Non-null.
   * @return A CSV dataset.
   * @throws IOException
   */
  public static CSV load(File file) throws IOException
  {
    return load(file, ',');
  }

  /**
   * Load a CSV dataset from file.
   *
   * @param file The file to read from. Non-null.
   * @param delimiter The delimiter separating values. Cannot be empty or null.
   * @return A CSV dataset.
   * @throws IOException
   */
  public static CSV load(File file, char delimiter) throws IOException
  {
    if (file == null)
      throw new IllegalArgumentException("file cannot be null");
    if (!file.exists())
      throw new FileNotFoundException("file does not exist");

    FileReader reader = new FileReader(file);
    BufferedReader br = new BufferedReader(reader);

    List<String> names = new ArrayList<String>();
    List<List<Object>> columns = new ArrayList<List<Object>>();
    for (String name : br.readLine().split(Character.toString(delimiter))) {
      names.add(name);
      columns.add(new ArrayList<Object>());
    }

    String line = br.readLine();

    while(line != null) {
      StringBuffer sb = new StringBuffer();
      List<String> values = new ArrayList<String>();
      for (int i = 0; i < line.length(); i++) {
        char c = line.charAt(i);
        if (c == delimiter) {
          values.add(sb.toString());
          sb.delete(0, sb.length());
        }
        else {
          sb.append(line.charAt(i));
        }

      }
      values.add(sb.toString());
      int c = 0;
      for (List<Object> column : columns) {
        try {
          column.add(Double.parseDouble(values.get(c++)));
        }
        catch (Exception e) {
          column.add(null);
        }

      }
      line = br.readLine();
    }

    br.close();
    return new CSV(names, columns);
  }

  /**
   * Set the delimiter which is used when saving.
   * @param delimiter A string to separate values. Cannot be empty or null.
   */
  public void setDelimiter(String delimiter)
  {
    if (delimiter == null)
      throw new IllegalArgumentException("delimiter cannot be null");
    if (delimiter.isEmpty())
      throw new IllegalArgumentException("delimiter cannot be empty");
    this.delimiter_ = delimiter;
  }

  /**
   * Save the dataset to a file.
   * @param file The destination file. Does not need to exist. Non-null.
   * @throws IOException
   */
  public void save(File file) throws IOException
  {
    if (file == null)
      throw new IllegalArgumentException("file cannot be null");

    StringBuilder sb = new StringBuilder();

    // Write the columns names
    for (String name : names_)
      sb.append(name + delimiter_);
    sb.replace(sb.length() - 1, sb.length(), "\n");

    for (int i = 0; i < columns_.get(0).size(); i++) {
      for (int j = 0; j < columns_.size(); j++) {
        Object value = columns_.get(j).get(i);
        sb.append(value != null ? value : "").append(delimiter_);
      }
      sb.replace(sb.length() - 1, sb.length(), "\n");
    }
    // Write to disk.
//    FileWriter fw = new FileWriter(file);
//    fw.write(sb.toString());
//    fw.close();

    Writer fstream = null;
    BufferedWriter out = null;

    fstream = new OutputStreamWriter(new FileOutputStream(file.getAbsolutePath()), StandardCharsets.UTF_8);
    out = new BufferedWriter(fstream);
    out.write(sb.toString());
    out.close();
    fstream.close();
  }

  /**
   * Get the names of each column in correct order.
   * @return An unmodifiable list of column names.
   */
  public List<String> getColumnNames() {
    return names_;
  }

  /**
   * Delete a column given its name. Has no effect if the named column does not exist.
   * @param name The name of the column to be deleted. Non-null.
   */
  public void deleteColumn(String name)
  {
    if (name == null)
      throw new IllegalArgumentException("name cannot be null");

    int i = 0;
    for (String s : names_) {
      if (s.equals(name)) {
        deleteColumn(i);
        return;
      }
      i++;
    }
  }

  /**
   * Delete a column given its index.
   * @param index The index of the column to be deleted.
   */
  public void deleteColumn(int index)
  {
    if (index < 0 || index >= columns_.size())
      throw new ArrayIndexOutOfBoundsException();
    names_.remove(index);
    columns_.remove(index);
  }

  /**
   * Get the values of the named column.
   * @param name The name of the column. Non-null.
   * @return A list of values. Null if column does not exist.
   */
  public List<Object> getColumn(String name)
  {
    if (name == null)
      throw new IllegalArgumentException("name cannot be null");

    int i = 0;
    for (String s : names_) {
      if (s.equals(name)) {
        return getColumn(i);
      }
      i++;
    }
    return null;
  }

  /**
   * Get the values of a specified column.
   * @param index The index of the column.
   * @return A list of values.
   */
  public List<Object> getColumn(int index)
  {
    if (index < 0 || index >= columns_.size())
      throw new ArrayIndexOutOfBoundsException();
    return columns_.get(index);
  }

  /**
   * Change the order of the columns. The new order is identical to the given column names.
   * @param order A list of column names. Must be of same length as there are columns. Non-null.
   */
  public void reorder(List<String> order)
  {
    if (order == null)
      throw new IllegalArgumentException("order cannot be null");
    if (order.size() != columns_.size())
      throw new IllegalArgumentException("the size of order must be equal to the number of columns");

    for (String name : order) {
      if (!names_.contains(name)) {
        throw new IllegalArgumentException("unknown column name: " + name);
      }
    }

    List<List<Object>> newList = new ArrayList<List<Object>>();
    for (String name : order) {
      newList.add(getColumn(name));
    }
    columns_ = newList;
    names_ = order;
  }

  /**
   * Change the name of a column.
   * @param oldName The name of column to rename. Non-null.
   * @param newName The new name of the specified column. Cannot be empty or null.
   */
  public void rename(String oldName, String newName)
  {
    if (oldName == null)
      throw new IllegalArgumentException("oldName cannot be null");
    if (newName == null)
      throw new IllegalArgumentException("newName cannot be null");
    if (newName.equals(""))
      throw new IllegalArgumentException("newName cannot be empty");

    for (int i = 0; i < names_.size(); i++) {
      if (oldName.equals(names_.get(i))) {
        names_.set(i, newName);
        return;
      }
    }
  }

}
