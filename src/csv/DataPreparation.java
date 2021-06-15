package csv;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


/**
 * Class for preparing raw log files to be used in machine learning. This includes converting the logs
 * to CSV files, dropping unwanted curves and renaming curves if they have a non-standard mnemonic.
 * This class in mainly intended to be used as a script.
 *
 * @author Karl Ostradt
 */
public final class DataPreparation
{

  public static void main(String[] args) throws IOException
  {
    File source = new File("path\\to\\source");
    File csvFolder = new File("C:path\\to\\csv_folder");
    File destinationFolder = new File("path\\to\\destination");

    start(source, csvFolder, destinationFolder);
  }

  /**
   * Rename non-standard mnemonics to a standard mnemonic if possible.
   *
   * @param curveNames The original curve names.
   * @return A list of the renames curve names.
   */
  static List<String> renameCurve(List<String> curveNames)
  {
    List<String> newList = new ArrayList<String>();

    for (String mnemonic : curveNames) {
      Mnemonic newMnemonic = Mnemonic.get(mnemonic);
      newList.add(newMnemonic == null ? mnemonic : newMnemonic.toString());
    }
    return newList;
  }

  /**
   * Starts the process of converting log files to CSV files. The CSV files are then transformed to conform to the same structure.
   * These transformations include dropping irrelevant curves and converting units if necessary.
   * @param source A directory containing log files in DLIS, LIS or LAS format to be bulk converted to CSV format. Sub folders will be accessed recursively.
   *               Can be null to prevent bulk conversion of log files.
   * @param csv A directory containing log files in CSV format. Sub folders will not be accessed.
   *            This is the destination folder when bulk conversion is enabled (i.e. <b>source</b> is not null). Not null.
   *            Can be null to prevent transforming the logs to the expected format used for machine learning in the LogAI project.
   * @param destination The directory to put transformed logs into. These logs are in CSV format of the expected format for the LogAI project.
   *                    Can be null to prevent transforming the raw logs in CSV format from the <b>csv</b> directory.
   */
  public static void start(File source, File csv, File destination)
  {
    if (csv == null)
      throw new IllegalArgumentException("csv cannot be null");
    if (!csv.isDirectory())
      throw new IllegalArgumentException("csv is not a directory");
    if (destination != null && !destination.isDirectory())
      throw new IllegalArgumentException("destination is not a directory");


    if (source != null) {
      if (!source.isDirectory())
        throw new IllegalArgumentException("source is not a directory");
      BulkConvert.bulkConvert(source, csv);
    }

    if (destination == null)
      return;

    for (File f : csv.listFiles()) {
      if (f.isDirectory())
        continue;
      try {
        read(f, destination.getPath());
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  private static void read(File f, String path) throws IOException
  {
    CSV dataset = CSV.load(f);
    List<String> originalNames = dataset.getColumnNames();
    List<String> standardNames = renameCurve(originalNames);
    Set<String> names = new HashSet<String>();

    for (int i = 0; i < originalNames.size(); i++) {
      dataset.rename(originalNames.get(i), standardNames.get(i));
      if (!names.add(originalNames.get(i)))
        return;
    }

    List<String> relevantCurves = relevantCurves();
    for (Object name : originalNames.toArray()) {
      if (!relevantCurves.contains(name.toString()))
        dataset.deleteColumn(name.toString());
    }
    if (dataset.getColumnNames().size() != relevantCurves.size())
      return;


    dataset.reorder(relevantCurves);
    try {
      dataset.save(new File(path + "\\" + f.getName()));
    } catch (Exception e) {
      e.printStackTrace();
    }

  }

  private static List<String> relevantCurves()
  {
    final List<String> set = new ArrayList<String>();
    set.add(Mnemonic.DEPTH.toString());
    set.add(Mnemonic.SONIC_COMPRESSIONAL.toString());
    set.add(Mnemonic.SONIC_SHEAR.toString());
    set.add(Mnemonic.BIT_SIZE.toString());
    set.add(Mnemonic.CALIPER.toString());
    set.add(Mnemonic.DENSITY.toString());
    set.add(Mnemonic.DENISTY_CORRECTION.toString());
    set.add(Mnemonic.GAMMA_RAY.toString());
    set.add(Mnemonic.NEUTRON_POROSITY.toString());
    set.add(Mnemonic.PHOTOELECTRIC_FACTOR.toString());
    set.add(Mnemonic.DEEP_RESISTIVITY.toString());
    set.add(Mnemonic.MEDIUM_RESISTIVITY.toString());

    return set;
  }

}
