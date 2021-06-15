package csv;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import no.petroware.logio.dlis.DlisCurve;
import no.petroware.logio.dlis.DlisFile;
import no.petroware.logio.dlis.DlisFileReader;
import no.petroware.logio.dlis.DlisFrame;
import no.petroware.logio.las.LasCurve;
import no.petroware.logio.las.LasFile;
import no.petroware.logio.las.LasFileReader;
import no.petroware.logio.lis.LisCurve;
import no.petroware.logio.lis.LisFile;
import no.petroware.logio.lis.LisFileReader;
import no.petroware.uom.Unit;
import no.petroware.uom.UnitManager;

/**
 * Class for bulk converting log files to CSV files.
 *
 * <br/><br/>
 * Log I/O is referenced as a separate source folder and not as a library from a JAR file.
 * As a benefit of this, the JAVADOC is automatically included.
 *
 * @author Karl Ostradt
 */
public final class BulkConvert
{
  private final static UnitManager um = UnitManager.getInstance();

  /**
   * Convert supported file types to CSV files.
   * @param folder The directory where the logs are stored. Must be a directory. Non-null.
   */
  static void bulkConvert(File folder, File destination)
  {
    if (folder == null)
      throw new IllegalArgumentException("folder cannot be null");
    if (!folder.isDirectory())
      throw new IllegalArgumentException("folder is not a directory");

    for (File f : folder.listFiles()) {
      if (f.isDirectory()) {
        bulkConvert(f, destination);
        continue;
      }

      String extension = f.getName().split("\\.")[1];

      if (extension.toUpperCase().equals("DLIS"))
        readFromDLIS(f, destination.getPath());
      else if (extension.toUpperCase().equals("LIS"))
        readFromLIS(f, destination.getPath());
      else if (extension.toUpperCase().equals("LAS"))
        readFromLAS(f, destination.getPath());
    }
  }

  private static void readFromLIS(File f, String path)
  {
    String fileName = f.getName().split("\\.")[0];

    LisFileReader reader = new LisFileReader(f);
    List<LisFile> lisFiles;
    try {
      lisFiles = reader.read(true, false, null);
      // Write the values in a csv format.
      int c = 0;
      for (LisFile lisFile : lisFiles) {
        String suffix = lisFiles.size() == 1 ? "" : "_" + c++;
        convertLisToCsv(lisFile, new File(path + "\\" + fileName + suffix + ".csv"));
      }
    } catch (IOException | InterruptedException e) {
      e.printStackTrace();
    }
  }

  private static void readFromDLIS(File f, String path)
  {
    String fileName = f.getName().split("\\.")[0];

    DlisFileReader reader = new DlisFileReader(f);
    List<DlisFile> dlisFiles;
    try {
      dlisFiles = reader.read(true, true, null);
      for (DlisFile dlisFile : dlisFiles) {
        List<DlisFrame> frames = dlisFile.getFrames();

        // Write the values in a csv format.
        int c = 0;
        for (DlisFrame frame : frames) {
          String suffix = (dlisFiles.size() == 1 && frames.size() == 1) ? "" : "_" + c++;
          convertDlisToCsv(frame, new File(path + "\\" + fileName + suffix + ".csv"));
        }
      }

    } catch (IOException | InterruptedException e) {
      System.out.println(f.getName());
      e.printStackTrace();
    }
  }

  private static void readFromLAS(File f, String path)
  {
    String fileName = f.getName().split("\\.")[0];

    LasFileReader reader = new LasFileReader(f);
    List<LasFile> lasFiles;
    try {
      lasFiles = reader.read(true);
      // Write the values in a csv format.
      int c = 0;
      for (LasFile lasFile : lasFiles) {
        String suffix = lasFiles.size() == 1 ? "" : "_" + c++;
        convertLasToCsv(lasFile, new File(path + "\\" + fileName + suffix + ".csv"));
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /**
   * Write the contents of a LisFile to disk using the CSV format.
   * @param lisFile The log to read from.
   * @param destination The destination file. Does not need to exist.
   * @throws IOException
   */
  public static void convertLisToCsv(LisFile lisFile, File destination) throws IOException
  {
    List<LisCurve> curves = lisFile.getCurves();
    List<Unit> units = new ArrayList<Unit>();
    List<String> curveNames = new ArrayList<String>();
    List<List<Object>> columns = new ArrayList<List<Object>>();

    for (LisCurve curve : curves) {
      String curveName = curve.getName();
      Unit from = um.findUnit(curve.getUnit());
      curveNames.add(curveName);
      units.add(from);

      Mnemonic curveDefaults = Mnemonic.get(curveName);
      Unit to = null;
      if (curveDefaults != null) {
        to = curveDefaults.getUnit();
      }
      List<Object> column = new ArrayList<Object>();

      for (int i = 0; i < curve.getNValues(); i++) {
        Object value = curve.getValue(i);
        if (value == null) {
          column.add(null);
          continue;
        }
        Double val = Double.parseDouble(value.toString());
        if (from != null && to != null && !from.equals(to))
          val = UnitManager.convert(from, to, val);
        column.add(val);
      }

      columns.add(column);
    }

    CSV csv = new CSV(curveNames, columns);
    csv.save(destination);
  }

  /**
   * Write the contents of a DlisFrame to disk using the CSV format.
   * @param frame The log to read from.
   * @param destination The destination file. Does not need to exist.
   * @throws IOException
   */
  public static void convertDlisToCsv(DlisFrame frame, File destination) throws IOException
  {
    List<DlisCurve> curves = frame.getCurves();
    List<Unit> units = new ArrayList<Unit>();
    List<String> curveNames = new ArrayList<String>();
    List<List<Object>> columns = new ArrayList<List<Object>>();

    for (DlisCurve curve : curves) {
      String curveName = curve.getName();
      Unit from = um.findUnit(curve.getUnit());
      curveNames.add(curveName);
      units.add(from);

      Mnemonic curveDefaults = Mnemonic.get(curveName);
      Unit to = null;
      if (curveDefaults != null) {
        to = curveDefaults.getUnit();
      }
      List<Object> column = new ArrayList<Object>();

      for (int i = 0; i < curve.getNValues(); i++) {
        Object value = curve.getValue(i);
        if (value == null) {
          column.add(null);
          continue;
        }
        Double val = Double.parseDouble(value.toString());
        if (from != null && to != null && !from.equals(to))
          val = UnitManager.convert(from, to, val);
        column.add(val);
      }

      columns.add(column);
    }

    CSV csv = new CSV(curveNames, columns);
    csv.save(destination);
  }

  /**
   * Write the contents of a LasFile to disk using the CSV format.
   * @param lasFile The log to read from.
   * @param destination The destination file. Does not need to exist.
   * @throws IOException
   */
  public static void convertLasToCsv(LasFile lasFile, File destination) throws IOException
  {
    List<LasCurve> curves = lasFile.getCurves();

    List<Unit> units = new ArrayList<Unit>();
    List<String> curveNames = new ArrayList<String>();
    List<List<Object>> columns = new ArrayList<List<Object>>();

    for (LasCurve curve : curves) {
      String curveName = curve.getName();
      Unit from = um.findUnit(curve.getUnit());
      curveNames.add(curveName);
      units.add(from);

      Mnemonic curveDefaults = Mnemonic.get(curveName);
      Unit to = null;
      if (curveDefaults != null) {
        to = curveDefaults.getUnit();
      }
      List<Object> column = new ArrayList<Object>();

      for (int i = 0; i < curve.getNValues(); i++) {
        Object value = curve.getValue(i);
        if (value == null) {
          column.add(null);
          continue;
        }
        Double val = Double.parseDouble(value.toString());
        if (from != null && to != null && !from.equals(to))
          val = UnitManager.convert(from, to, val);
        column.add(val);
      }

      columns.add(column);
    }

    CSV csv = new CSV(curveNames, columns);
    csv.save(destination);
  }

}
