package csv;

import no.petroware.uom.Unit;
import no.petroware.uom.UnitManager;
/**
 * Enumerator for mapping different mnemonics of same meaning to have the same mnemonic.
 * For example, both 'GR' and 'HGR' is mapped to 'GR'.
 *
 * @author Karl Ostradt
 */
enum Mnemonic
{
  DEPTH("DEPTH", new String[] {"DEPTH", "Depth", "DEPT"}, UnitManager.getInstance().findUnit("m")),
  SONIC_COMPRESSIONAL("AC", new String[] {"AC", "DT24", "DTC", "HDT"}, UnitManager.getInstance().findUnit("us/ft")),
  SONIC_SHEAR("ACS", new String[] {"ACS", "DTS", "DT4S"}, UnitManager.getInstance().findUnit("us/ft")),
  BIT_SIZE("BS", new String[] {"BS", "HBS"}, UnitManager.getInstance().findUnit("in")),
  CALIPER("CALI", new String[] {"CALI", "CAL", "HCAL", "HCAL_1", "HCALI", "RSO8"}, UnitManager.getInstance().findUnit("in")),
  DENSITY("DEN", new String[] {"DEN", "HDEN", "HRHO", "RHO8"}, UnitManager.getInstance().findUnit("g/cm3")),
  DENISTY_CORRECTION("DENC", new String[] {"DENC", "HCOR", "HDRH"}, UnitManager.getInstance().findUnit("g/cm3")),
  GAMMA_RAY("GR", new String[] {"GR", "HGR", "EHGR", "HDRHO", "HNPHI", "HRHOB"}, UnitManager.getInstance().findUnit("gAPI")),
  NEUTRON_POROSITY("NEU", new String[] {"NEU", "HCN", "HNPO", "HPHI"}, UnitManager.getInstance().findUnit("m")),
  PHOTOELECTRIC_FACTOR("PEF", new String[] {"PEF", "PE", "HPEF"}, null),
  DEEP_RESISTIVITY("RDEP", new String[] {"RDEP", "HDR", "HRLD"}, UnitManager.getInstance().findUnit("ohm.m")),
  MEDIUM_RESISTIVITY("RMED", new String[] {"RMED", "HRM", "HRLS"}, UnitManager.getInstance().findUnit("ohm.m")),
  RATE_OF_PENETRATION("ROP", new String[] {"ROP"}, UnitManager.getInstance().findUnit("m/h")),
  SHALLOW_RESISTIVITY("RSHA", new String[] {"RSHA"}, UnitManager.getInstance().findUnit("ohm.m")),
  MICRO_RESISTIVITY("HRS", new String[] {"HRS", "HMIN", "HMNO", "RXO8", "RMIC"}, UnitManager.getInstance().findUnit("ohm.m")),
  POTASSIUM("K", new String[] {"K"}, UnitManager.getInstance().findUnit("%")),
  THORIUM("TH", new String[] {"TH"}, UnitManager.getInstance().findUnit("ppm")),
  URANIUM("U", new String[] {"U"}, UnitManager.getInstance().findUnit("ppm"));

  private final String mnemonic_;
  private final String[] aliases_;
  private final Unit unit_;

  private Mnemonic(String mnemonic, String[] aliases, Unit unit)
  {
    assert aliases != null : "aliases cannot be null";
    mnemonic_ = mnemonic;
    aliases_ = aliases;
    unit_ = unit;
  }

  static Mnemonic get(String alias)
  {
    if (alias == null)
      throw new IllegalArgumentException("alias cannot be null");

    for (Mnemonic m : Mnemonic.values()) {
      for (String s : m.aliases_) {
        if (s.equals(alias))
          return m;
      }
    }
    return null;
  }

  Unit getUnit()
  {
    return unit_;
  }

  @Override
  public String toString()
  {
    return mnemonic_;
  }
}
